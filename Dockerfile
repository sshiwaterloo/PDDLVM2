# PDDLVM container (CPU).
#
# Build:
#   docker build -t pddlvm:2.0.0 .
#
# Run an experiment (mount a workspace for artifacts):
#   docker run --rm -v "$PWD/artifacts:/work/artifacts" \
#       pddlvm:2.0.0 python run.py configs/ve_prior_default.json --steps 200
#
# A GPU variant should install the CUDA-enabled tensorflow wheel instead of
# the CPU one.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /work

# tensorflow (CPU) is a heavyweight dependency; install the package's
# dependencies explicitly first so edits to the source do not invalidate
# the layer cache.
RUN pip install --upgrade pip \
    && pip install "tensorflow>=2.13" "tensorflow-probability>=0.22" \
                   "numpy>=1.26" "scipy>=1.10" "matplotlib>=3.7"

COPY pyproject.toml README.md CHANGELOG.md LICENSE ./
COPY pddlvm ./pddlvm
COPY configs ./configs
COPY run.py ./

RUN pip install --no-deps .

CMD ["python", "run.py", "configs/ve_prior_default.json"]
