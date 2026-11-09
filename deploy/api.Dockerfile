# Production api, worker, beat and sandbox image for the self-hosted stack (deploy/stack.yml).
# BASE is the image built from the root Dockerfile. Boto3 has no env setting for path-style
# S3, which MinIO needs, and a Git-backed Portainer stack cannot bind-mount a repo file.
ARG BASE=overmind-dev
FROM ${BASE}
COPY deploy/minio-aws-config /code/.aws-minio/config
