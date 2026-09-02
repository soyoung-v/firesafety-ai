"""/m_noUpload.php로 HTTP GET 전송. 표준 라이브러리만 사용(신규 의존성 추가 없음)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

INGEST_PATH = "/m_noUpload.php"


class FrameSendError(Exception):
    def __init__(self, status_code: int | None, message: str):
        super().__init__(message)
        self.status_code = status_code


# 프레임 파라미터 1개를 GET 쿼리스트링으로 전송하고 응답 JSON을 반환
def send_frame(base_url: str, params: dict[str, str], timeout_s: float = 5.0) -> dict:
    url = f"{base_url.rstrip('/')}{INGEST_PATH}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise FrameSendError(e.code, f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise FrameSendError(None, f"연결 실패: {e.reason}") from e
    except TimeoutError as e:
        # 연결 후 응답 read 단계의 타임아웃은 urllib.error.URLError로 감싸지지 않고
        # 순수 socket.timeout(=TimeoutError)로 그대로 올라온다 - 별도로 잡아야 한다.
        raise FrameSendError(None, f"응답 타임아웃({timeout_s}s)") from e
