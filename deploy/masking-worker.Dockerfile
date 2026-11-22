# The masking worker: the api image plus CPU-only torch, transformers and two pinned models,
# both stored as int8 so the worker never holds fp32 weights in memory.
# Build from the repository root, after the api image:
#   docker build -f deploy/masking-worker.Dockerfile --build-arg BASE=overmind-api -t overmind-masking .
# Weights are fetched once, at build time, at exact commits, quantised in a build stage that is
# not shipped, and the worker runs offline.
#   - LatentAILabs/chitragupta-pii-indic: names and Indian identifiers. Apache-2.0.
#   - knowledgator/gliner-x-base: zero-shot GLiNER for medical, place, relation and lender words
#     no list holds (mT5 encoder, multilingual). Apache-2.0 (model card).
ARG BASE=overmind-dev
ARG UV_VERSION=0.11.24

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${BASE} AS runtime
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
# The extra's wheels only; the project itself is already installed in /code/.venv.
RUN uv sync --frozen --no-dev --no-editable --no-install-project --extra masking \
    && rm -f /usr/local/bin/uv

FROM runtime AS quantise
# The defaults are the revisions the code was reviewed against.
ARG MASKING_MODEL_REPO=LatentAILabs/chitragupta-pii-indic
ARG MASKING_MODEL_REVISION=ad42bdadf9f4ec9884c4bdbf3c1b710ee52aa795
ARG MASKING_CONTEXT_REPO=knowledgator/gliner-x-base
ARG MASKING_CONTEXT_REVISION=a6c7a8fbfe2066c68d12c0480418504fde6d91d1
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${MASKING_MODEL_REPO}', revision='${MASKING_MODEL_REVISION}', \
local_dir='/fp32/chitragupta', \
allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','LICENSE','NOTICE']); \
snapshot_download('${MASKING_CONTEXT_REPO}', revision='${MASKING_CONTEXT_REVISION}', \
local_dir='/fp32/gliner', \
allow_patterns=['gliner_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json'])" \
    && cd /code \
    && python -m overbae.services.masking.quantise chitragupta /fp32/chitragupta /models/chitragupta \
    && python -m overbae.services.masking.quantise gliner /fp32/gliner /models/gliner \
    && echo "${MASKING_MODEL_REVISION}" > /models/chitragupta/.revision \
    && echo "${MASKING_CONTEXT_REVISION}" > /models/gliner/.revision

FROM runtime
ARG MASKING_MODEL_REVISION=ad42bdadf9f4ec9884c4bdbf3c1b710ee52aa795
ARG MASKING_CONTEXT_REVISION=a6c7a8fbfe2066c68d12c0480418504fde6d91d1
COPY --from=quantise /models /models
# Two threads and two malloc arenas keep the worker's peak under 3.5 GB (measured: 3.3 GB with both
# models; ten threads and default arenas grew it past 4 GB).
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MASKING_ENABLED=1 \
    MASKING_MODEL_DIR=/models/chitragupta \
    MASKING_MODEL_REVISION=${MASKING_MODEL_REVISION} \
    MASKING_CONTEXT_DIR=/models/gliner \
    MASKING_CONTEXT_REVISION=${MASKING_CONTEXT_REVISION}
