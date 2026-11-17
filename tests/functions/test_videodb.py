from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pixeltable as pxt

from ..utils import (
    get_video_files,
    rerun_on_network_error,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


@pytest.fixture
def mock_connection(init_env: None, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    skip_test_if_not_installed('videodb')
    import videodb  # type: ignore[import-untyped]

    video = MagicMock(id='m-test')
    video.get_transcript.return_value = [
        {'start': 0, 'end': 1.5, 'text': 'Hello there.'},
        {'start': 1.5, 'end': 3, 'text': 'General Kenobi.'},
    ]
    video.index_scenes.return_value = 'scene-index'
    video.get_scene_index.return_value = [{'start': 0, 'end': 10, 'description': 'A cooking show.', 'metadata': {}}]
    collection = MagicMock()
    collection.upload.return_value = video
    collection.get_video.return_value = video
    conn = MagicMock()
    conn.get_collection.return_value = collection
    monkeypatch.setenv('VIDEODB_API_KEY', 'test-key')
    monkeypatch.setattr(videodb, 'connect', MagicMock(return_value=conn))
    return conn


class TestVideoDBLocal:
    def test_udfs(self, uses_db: None, mock_connection: MagicMock) -> None:
        import videodb

        from pixeltable.functions.videodb import describe_scenes, transcribe, upload

        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(videodb_id=upload(t.video, collection_id='c-test'))
        t.add_computed_column(transcript=transcribe(t.videodb_id, collection_id='c-test', language_code='en'))
        t.add_computed_column(
            scenes=describe_scenes(
                t.videodb_id, collection_id='c-test', extraction_type='time', extraction_config={'time': 10}
            )
        )
        video_file = get_video_files()[0]
        validate_update_status(t.insert(video=video_file), 1)

        res = t.collect()
        assert res['videodb_id'][0] == 'm-test'
        assert res['transcript'][0] == [
            {'start': 0.0, 'end': 1.5, 'text': 'Hello there.'},
            {'start': 1.5, 'end': 3.0, 'text': 'General Kenobi.'},
        ]
        assert res['scenes'][0] == [{'start': 0.0, 'end': 10.0, 'description': 'A cooking show.'}]

        videodb.connect.assert_called_with(api_key='test-key')
        mock_connection.get_collection.assert_called_with('c-test')
        collection = mock_connection.get_collection.return_value
        collection.upload.assert_called_once_with(file_path=video_file)
        collection.get_video.assert_called_with('m-test')
        video = collection.get_video.return_value
        video.generate_transcript.assert_called_once_with(language_code='en')
        video.get_transcript.assert_called_once_with(segmenter='sentence')
        video.index_scenes.assert_called_once_with(extraction_type='time', extraction_config={'time': 10}, prompt=None)
        video.get_scene_index.assert_called_once_with('scene-index')
        # every call closes the connection it opened
        assert mock_connection.session.close.call_count == mock_connection.get_collection.call_count == 3

    def test_upload_without_video_id(self, uses_db: None, mock_connection: MagicMock) -> None:
        from pixeltable.functions.videodb import upload

        mock_connection.get_collection.return_value.upload.return_value = SimpleNamespace(id='a-test')
        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(videodb_id=upload(t.video))
        t.insert(video=get_video_files()[0], on_error='ignore')
        res = t.select(t.videodb_id, error=t.videodb_id.errormsg).collect()
        assert res['videodb_id'][0] is None
        assert 'did not return a video ID' in res['error'][0]


@pytest.mark.expensive
@pytest.mark.skip_merge_queue
@rerun_on_network_error()
class TestVideoDB:
    def test_transcribe_and_describe_scenes(self, uses_db: None) -> None:
        skip_test_if_not_installed('videodb')
        skip_test_if_no_client('videodb')
        from pixeltable.functions.videodb import describe_scenes, transcribe, upload

        t = pxt.create_table('test_tbl', {'video': pxt.Video})
        t.add_computed_column(videodb_id=upload(t.video))
        t.add_computed_column(transcript=transcribe(t.videodb_id))
        t.add_computed_column(
            scenes=describe_scenes(t.videodb_id, extraction_type='time', extraction_config={'time': 10})
        )
        validate_update_status(t.insert(video=get_video_files(extension='.webm')[0]), 1)
        res = t.collect()
        assert res['videodb_id'][0].startswith('m-')
        assert isinstance(res['transcript'][0], list)
        scenes = res['scenes'][0]
        assert len(scenes) > 0
        assert all(len(scene['description']) > 0 for scene in scenes)
