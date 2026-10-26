FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin:$PATH"
ENV UV_PYTHON=/usr/local/bin/python

WORKDIR /code

COPY pyproject.toml uv.lock README.md /code/
COPY ./src /code/src

RUN uv sync --frozen

COPY ./app /code/app
COPY cnn_model.pth /code/cnn_model.pth
COPY generator.pth /code/generator.pth

RUN uv pip install \
    --python /code/.venv/bin/python \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl

CMD ["/code/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
