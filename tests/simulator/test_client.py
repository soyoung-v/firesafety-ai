import io
import json
import urllib.error
import urllib.request

import pytest

from simulator import client


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_send_frame_builds_get_url_with_query_params(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        return _FakeResponse(json.dumps({"message": "ok"}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = client.send_frame("http://localhost:8080", {"m_no": "00001", "volt": "224"})

    assert captured["method"] == "GET"
    assert captured["url"].startswith("http://localhost:8080/m_noUpload.php?")
    assert "m_no=00001" in captured["url"]
    assert "volt=224" in captured["url"]
    assert result == {"message": "ok"}


def test_send_frame_strips_trailing_slash_from_base_url(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return _FakeResponse(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client.send_frame("http://localhost:8080/", {"m_no": "00001"})

    assert captured["url"] == "http://localhost:8080/m_noUpload.php?m_no=00001"


def test_send_frame_raises_frame_send_error_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", None, io.BytesIO(b'{"message":"error"}')
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(client.FrameSendError) as exc_info:
        client.send_frame("http://localhost:8080", {"m_no": "00001"})
    assert exc_info.value.status_code == 400


def test_send_frame_raises_frame_send_error_on_connection_failure(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(client.FrameSendError) as exc_info:
        client.send_frame("http://localhost:8080", {"m_no": "00001"})
    assert exc_info.value.status_code is None


def test_send_frame_raises_frame_send_error_on_bare_read_timeout(monkeypatch):
    # 응답 read 단계 타임아웃은 urllib.error.URLError로 감싸지지 않고 순수 TimeoutError로 올라온다
    def fake_urlopen(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(client.FrameSendError) as exc_info:
        client.send_frame("http://localhost:8080", {"m_no": "00001"})
    assert exc_info.value.status_code is None
