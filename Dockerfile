FROM python:3.10-slim-bookworm

ARG CROPPER_GITHUB_USER=la60312
ARG CROPPER_GITHUB_REPO=GestaltEngine-FaceCropper
ARG CROPPER_GITHUB_BRANCH=research_platform

ARG CROPPER_PORT=5000

ENV HOME=/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl vim\
        libglib2.0-0 libsm6 libxrender1 libxext6\
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -U --no-create-home -u 9999 -s /bin/bash app && \
    mkdir -p /app && chown -R app:app /app

USER app

# Option 1: Fetch from GitHub using BuildKit secrets (secure method)
# WORKDIR /app
# RUN --mount=type=secret,id=github_token,uid=9999 \
#     CROPPER_GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
#     AUTH=$([ -n "${CROPPER_GITHUB_TOKEN}" ] && echo "${CROPPER_GITHUB_USER}:${CROPPER_GITHUB_TOKEN}@" || echo "") && \
#     git clone --branch ${CROPPER_GITHUB_BRANCH} --single-branch https://${AUTH}github.com/${CROPPER_GITHUB_USER}/${CROPPER_GITHUB_REPO}.git && \
#     cd ${CROPPER_GITHUB_REPO}

# Option 2: Copy project files from local directory
# Install dependencies before copying source for better layer caching.
# torch/torchvision are large; this avoids re-downloading on source changes.
WORKDIR /app/${GITHUB_REPO}
COPY ./requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY --chown=app:app . .

ENV CROPPER_HOST=0.0.0.0
ENV CROPPER_PORT=${CROPPER_PORT}

EXPOSE ${CROPPER_PORT}
CMD ["python", "cropper_server.py"]
