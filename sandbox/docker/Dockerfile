FROM python:3.12-slim

RUN groupadd --gid 65532 sandbox \
    && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin sandbox

USER 65532:65532
WORKDIR /tmp
