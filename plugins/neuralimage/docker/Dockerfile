FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY plugins ./plugins
RUN uv sync --frozen --package neuralimage --extra server --no-dev
ENV PATH="/app/.venv/bin:$PATH" NEURALIMAGE_DATA_DIR=/data PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 neuralimage && mkdir /data && chown neuralimage /data
USER neuralimage
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/v1/capabilities', timeout=3)"
ENTRYPOINT ["neuralimage-server"]
CMD ["serve"]
