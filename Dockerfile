# Image for the Milestone 3 experiments: the project code and the exact Python dependencies, nothing else.
# The data (processed_data/), the MLflow store (mlruns/), the reports and any credentials are NOT in the image: they are mounted
# when the container runs (see the README). The commit that produced the image is passed in as a build argument because .git is
# not part of the build context (.dockerignore).
#
# Two variants from this one file (the CPU image is the default):
#   CPU:  docker build -t pm25-m3:cpu --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
#   GPU:  docker build -t pm25-m3:gpu --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg TORCH_INDEX=https://pypi.org/simple .
#         (the GPU image installs the CUDA build of PyTorch from PyPI; it runs on a machine with an NVIDIA driver and `--gpus all`)
#
# Run an experiment (processed data read-only; run records and reports written back to the host):
#   docker run --rm -v "$PWD/processed_data:/app/processed_data:ro" -v "$PWD/mlruns:/app/mlruns" -v "$PWD/reports:/app/reports" \
#       pm25-m3:cpu src/train.py --config configs/xgboost_default_feature_set_a.yaml
ARG PYTHON_VERSION=3.14.2
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# PyTorch first, from the chosen index (CPU build by default), then everything else from the pinned requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY metadata/ metadata/

ENTRYPOINT ["python"]
CMD ["src/train.py", "--help"]
