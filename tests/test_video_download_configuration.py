from __future__ import annotations

from pathlib import Path

import yt_dlp

from camel.toolkits import video_download_toolkit
from camel.toolkits.video_download_toolkit import VideoDownloaderToolkit


class _FakeYoutubeDL:
    options = None

    def __init__(self, options):
        type(self).options = options

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def extract_info(self, url, download):
        assert url == "https://www.youtube.com/watch?v=test"
        assert download is True
        return {"title": "test", "ext": "mp4"}

    def prepare_filename(self, info):
        return "/tmp/test.mp4"


def test_youtube_download_uses_cookie_and_configured_player(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "OWL_VIDEO_COOKIES_FROM_BROWSER",
        "chrome+basictext:/profiles/youtube/Default",
    )
    monkeypatch.setenv("OWL_VIDEO_YOUTUBE_PLAYER_CLIENT", "mweb")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)

    toolkit = VideoDownloaderToolkit(download_directory=str(tmp_path))
    assert (
        toolkit.download_video("https://www.youtube.com/watch?v=test")
        == "/tmp/test.mp4"
    )

    options = _FakeYoutubeDL.options
    assert options["cookiesfrombrowser"] == (
        "chrome",
        "/profiles/youtube/Default",
        "BASICTEXT",
        None,
    )
    assert options["extractor_args"] == {
        "youtube": {"player_client": ["mweb"]}
    }
    assert options["format"].startswith("best[protocol^=m3u8]")
    assert "force_generic_extractor" not in options


def test_non_youtube_download_keeps_normal_format(tmp_path, monkeypatch):
    monkeypatch.delenv("OWL_VIDEO_COOKIES_FROM_BROWSER", raising=False)
    monkeypatch.delenv("OWL_VIDEO_YOUTUBE_PLAYER_CLIENT", raising=False)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)
    monkeypatch.setattr(
        _FakeYoutubeDL,
        "extract_info",
        lambda self, url, download: {"title": "test", "ext": "mp4"},
    )

    toolkit = VideoDownloaderToolkit(download_directory=str(tmp_path))
    toolkit.download_video("https://videos.example.test/movie")

    options = _FakeYoutubeDL.options
    assert options["format"] == "bestvideo+bestaudio/best"
    assert "extractor_args" not in options
    assert "force_generic_extractor" not in options


def test_default_download_directory_is_under_owl_tmp(monkeypatch):
    monkeypatch.delenv("OWL_TMP_DIR", raising=False)
    toolkit = VideoDownloaderToolkit()

    owl_root = Path(video_download_toolkit.__file__).resolve().parents[2]
    assert toolkit._download_directory.parent == owl_root / "tmp"
    assert toolkit._download_directory.name.startswith("video-download-")


def test_default_download_directory_honors_owl_tmp_dir(
    tmp_path, monkeypatch
):
    custom_root = tmp_path / "owl-tmp"
    monkeypatch.setenv("OWL_TMP_DIR", str(custom_root))
    toolkit = VideoDownloaderToolkit()

    assert toolkit._download_directory.parent == custom_root
