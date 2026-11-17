"""
Pixeltable UDFs
that wrap various endpoints from the VideoDB API. In order to use them, you must
first `pip install videodb` and configure your VideoDB API key, by setting the `VIDEODB_API_KEY` environment
variable or `videodb.api_key` in the Pixeltable config file.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Iterator, Literal, TypedDict

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime

if TYPE_CHECKING:
    from videodb.collection import Collection  # type: ignore[import-untyped]


class _VideoDBClient:
    """Opens a new videodb Connection for each call: a Connection keeps per-request polling state, so concurrent
    calls can't share one."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @contextlib.contextmanager
    def collection(self, collection_id: str) -> Iterator['Collection']:
        import videodb  # type: ignore[import-untyped]

        conn = videodb.connect(api_key=self.api_key)
        try:
            yield conn.get_collection(collection_id)
        finally:
            conn.session.close()


@register_client('videodb', credential_param='api_key')
def _(api_key: str) -> _VideoDBClient:
    return _VideoDBClient(api_key)


def _videodb_client() -> _VideoDBClient:
    return get_runtime().get_client('videodb')


class TranscriptSegment(TypedDict):
    """One timestamped segment returned by `transcribe()`."""

    start: float
    """Start of the segment in seconds."""
    end: float
    """End of the segment in seconds."""
    text: str


class SceneDescription(TypedDict):
    """One scene returned by `describe_scenes()`."""

    start: float
    """Start of the scene in seconds."""
    end: float
    """End of the scene in seconds."""
    description: str


@pxt.udf(is_deterministic=False, resource_pool='request-rate:videodb')
async def upload(video: pxt.Video, *, collection_id: str = 'default') -> str:
    """
    Uploads a video to a VideoDB collection and returns its VideoDB video ID.

    The other UDFs in this module take that ID, so each video is uploaded once. Equivalent to the VideoDB
    `Collection.upload()` method:
    <https://docs.videodb.io/api-reference/collections/upload_to_collection>

    Request throttling:
    Applies the rate limit set in the config (section `videodb`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video: The video to upload.
        collection_id: The VideoDB collection to upload to.

    Returns:
        The VideoDB video ID, such as `m-...`.

    Examples:
        Add a computed column with the VideoDB ID of each video in the column `tbl.video`:

        >>> tbl.add_computed_column(videodb_id=upload(tbl.video))
    """
    Env.get().require_package('videodb')
    client = _videodb_client()

    def run() -> str:
        with client.collection(collection_id) as coll:
            media = coll.upload(file_path=video)
            if media is None or not media.id.startswith('m-'):
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                    'VideoDB did not return a video ID for the upload.',
                    provider='videodb',
                )
            return media.id

    return await asyncio.to_thread(run)


@pxt.udf(resource_pool='request-rate:videodb')
async def transcribe(
    video_id: str,
    *,
    collection_id: str = 'default',
    language_code: str | None = None,
    segmenter: Literal['word', 'sentence'] = 'sentence',
) -> list[TranscriptSegment]:
    """
    Transcribes the speech in a VideoDB video and returns the timestamped transcript.

    Equivalent to the VideoDB `Video.generate_transcript()` and `Video.get_transcript()` methods:
    <https://docs.videodb.io/api-reference/videos/transcription/index>

    VideoDB keeps the transcript after the first call, so later calls for the same video return it without
    transcribing again.

    Request throttling:
    Applies the rate limit set in the config (section `videodb`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        language_code: The spoken language as an ISO 639-1 code, such as `en` or `fr`. If not specified,
            VideoDB uses `en_us`.
        segmenter: Whether to return one segment per `word` or per `sentence`.

    Returns:
        A list of transcript segments.

    Examples:
        Add a computed column with the transcript of each uploaded video:

        >>> tbl.add_computed_column(videodb_id=upload(tbl.video))
        ... tbl.add_computed_column(transcript=transcribe(tbl.videodb_id))
    """
    Env.get().require_package('videodb')
    client = _videodb_client()

    def run() -> list[dict[str, Any]]:
        with client.collection(collection_id) as coll:
            video = coll.get_video(video_id)
            video.generate_transcript(language_code=language_code)
            return video.get_transcript(segmenter=segmenter)

    segments = await asyncio.to_thread(run)
    return [TranscriptSegment(start=float(s['start']), end=float(s['end']), text=s['text']) for s in segments]


@pxt.udf(is_deterministic=False, resource_pool='request-rate:videodb')
async def describe_scenes(
    video_id: str,
    *,
    collection_id: str = 'default',
    prompt: str | None = None,
    extraction_type: Literal['shot', 'time'] = 'shot',
    extraction_config: dict[str, Any] | None = None,
) -> list[SceneDescription]:
    """
    Splits a VideoDB video into scenes and describes each scene with a vision-language model.

    Equivalent to the VideoDB `Video.index_scenes()` and `Video.get_scene_index()` methods:
    <https://docs.videodb.io/pages/understand/legacy-indexing-search/create-an-index>

    Each call creates a new scene index on the video.

    Request throttling:
    Applies the rate limit set in the config (section `videodb`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install videodb`

    Args:
        video_id: The VideoDB video ID, as returned by `upload()`.
        collection_id: The VideoDB collection that contains the video.
        prompt: Instructions for the scene descriptions. If not specified, VideoDB uses its default prompt.
        extraction_type: Whether to split the video at camera `shot` changes or at fixed `time` intervals.
        extraction_config: Additional extraction options, such as `threshold` for `shot` or `time` (the interval
            in seconds) for `time`, and `frame_count` for either.

    Returns:
        A list of scene descriptions.

    Examples:
        Add a computed column with scene descriptions every 10 seconds of each uploaded video:

        >>> tbl.add_computed_column(
        ...     scenes=describe_scenes(
        ...         tbl.videodb_id,
        ...         prompt='Describe the people and objects in the scene.',
        ...         extraction_type='time',
        ...         extraction_config={'time': 10},
        ...     )
        ... )
    """
    Env.get().require_package('videodb')
    client = _videodb_client()

    def run() -> list[dict[str, Any]]:
        with client.collection(collection_id) as coll:
            video = coll.get_video(video_id)
            scene_index_id = video.index_scenes(
                extraction_type=extraction_type,
                extraction_config=extraction_config if extraction_config is not None else {},
                prompt=prompt,
            )
            if scene_index_id is None:
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                    f'VideoDB did not return a scene index for video {video_id}.',
                    provider='videodb',
                )
            return video.get_scene_index(scene_index_id) or []

    records = await asyncio.to_thread(run)
    return [
        SceneDescription(start=float(r['start']), end=float(r['end']), description=r['description']) for r in records
    ]


__all__ = ['SceneDescription', 'TranscriptSegment', 'describe_scenes', 'transcribe', 'upload']


def __dir__() -> list[str]:
    return __all__
