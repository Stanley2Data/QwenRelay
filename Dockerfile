# QwenRelay local source build recipe. Fresh image/GPU startup remains unvalidated.
FROM nvidia/cuda:13.0.3-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev git patch ca-certificates curl \
      build-essential cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN bash scripts/install-qwenrelay.sh
RUN mkdir -p /cache /models
ENV HOME=/cache MODELS_ROOT=/models API_KEY_FILE=/run/secrets/qwenrelay_api_key BIND_HOST=0.0.0.0
EXPOSE 18020
ENTRYPOINT ["bash", "scripts/start-qwenrelay.sh"]
