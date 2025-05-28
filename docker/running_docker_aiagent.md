# Docker Guide for `aiagent-sglang`

This document explains how to build, run, and test the Docker image for `aiagent-sglang`.

---

## 1. Build the Docker Image

Use the following command to build the image from `Dockerfile.aiagent`:

```bash
docker build --no-cache -f Dockerfile.aiagent -t aiagent-sglang:latest .
```

- `-f Dockerfile.aiagent`: Specifies the custom Dockerfile.
- `-t aiagent-sglang:latest`: Tags the image as `latest`.

---


## 2. Run the Docker Container with Compose

Use the following command to run the `aiagent-sglang` server with GPU support using the `aiagent_server_compose.yaml` file:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose -f aiagent_server_compose.yaml up
```

- `CUDA_VISIBLE_DEVICES=0,1,2,3`: Restricts the container to use GPUs 0–3.
- `-f aiagent_server_compose.yaml`: Specifies the Docker Compose file.
- `up`: Starts the container and streams logs to the terminal.

---