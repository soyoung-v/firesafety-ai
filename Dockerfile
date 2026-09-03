# firesafety-ai(FastAPI 추론 서비스) 이미지. 학습/Dataset Generator 전체는 이 이미지에 포함하지 않는다 -
# 이미지에는 추론에 실제로 필요한 것만 담는다: app/, training/features/·training/models/(추론 코드
# 재사용, api-contract.md 5절), artifacts/(학습된 모델 파일). 모델은 이 빌드 과정에서 재학습하지 않고
# 저장소에 이미 커밋된 artifacts/를 그대로 사용한다.
#
# training/scenario/와 simulator/도 포함한다 - Continuous Demo Simulator(deploy/systemd 참고)가
# `docker compose run --rm --no-deps ai-service python -m simulator.continuous_run ...`로 이 이미지를
# 그대로 재사용해 별도 EC2 host venv나 소스 clone 없이 실행되기 때문이다(source of truth를 이 이미지
# 하나로 유지). CMD는 여전히 FastAPI 서빙이고, 이 코드는 `docker compose run`으로 명령을 오버라이드할
# 때만 쓰인다 - 상시 서빙 컨테이너의 동작에는 영향이 없다.
#
# 개발 Mac(Apple Silicon)에서 이미지를 만들어 EC2(x86_64)로 옮길 때만
# `docker buildx build --platform linux/amd64 ...`로 빌드할 것.

FROM python:3.11-slim AS build
WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

RUN addgroup --system firesafety && adduser --system --ingroup firesafety firesafety

COPY --from=build /install /usr/local

COPY app ./app
COPY training/__init__.py ./training/__init__.py
COPY training/features ./training/features
COPY training/models ./training/models
COPY training/scenario ./training/scenario
COPY artifacts ./artifacts
COPY simulator ./simulator

USER firesafety

EXPOSE 8000

# uvicorn 자체 헬스체크 엔드포인트는 없으므로 Python 표준 라이브러리로 /health를 호출한다.
# localhost가 아니라 127.0.0.1을 명시한다 - /etc/hosts는 localhost를 127.0.0.1과 ::1 둘 다로
# 매핑하는데, 이 이미지는 IPv6가 비활성화된 네트워크에서 돌 수 있어(disable_ipv6=1) urllib이
# IPv6로 폴백을 시도하면 "Errno 99 Cannot assign requested address"로 실패 원인이 가려진다.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
