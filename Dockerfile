FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config ./config
# State and briefings live on mounted volumes; secrets come from --env-file.
ENV DATA_DIR=/data OUTPUT_DIR=/output
VOLUME ["/data", "/output"]
USER nobody
ENTRYPOINT ["competitive-briefing"]
