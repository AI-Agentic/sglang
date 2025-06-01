# Build stage
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone repository with specific branch
RUN git clone -b tile_pruning https://github.com/AI-Agentic/sglang

# Install dependencies
WORKDIR /build/sglang
RUN pip install --upgrade pip
RUN pip install -e "python[all]"
RUN pip install pyarmor

# Copy your license file into the container
COPY pyarmor-regfile-9285.zip /tmp/pyarmor-license.zip

# Register the license
RUN pyarmor reg /tmp/pyarmor-license.zip

# Clean up the license file for security
RUN rm /tmp/pyarmor-license.zip

# Obfuscate the code
RUN pyarmor gen -O dist-obf -r -i python/sglang/
# RUN pyarmor gen -O dist-obf -r -i python/sglang_router/

# Copy obfuscated files back
RUN cp -r dist-obf/python/sglang/* python/sglang/
# RUN cp -r dist-obf/python/sglang_router/* python/sglang_router/

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire sglang directory with obfuscated code
COPY --from=builder /build/sglang /app/sglang

# Install the package
WORKDIR /app/sglang
RUN pip install --upgrade pip
RUN pip install -e "python[all]"
RUN pip install sglang_router

# Create wrapper scripts
RUN echo '#!/usr/bin/env python3\nimport sys\nsys.path.insert(0, "/app/sglang/sgl-router/py_src")\nfrom sglang_router.launch_router import main\nif __name__ == "__main__":\n    sys.exit(main())' > /usr/local/bin/router && \
    chmod +x /usr/local/bin/router

RUN echo '#!/usr/bin/env python3\nimport sys\nsys.path.insert(0, "/app/sglang/sgl-router/py_src")\nfrom sglang_router.launch_server import main\nif __name__ == "__main__":\n    sys.exit(main())' > /usr/local/bin/server && \
    chmod +x /usr/local/bin/server

# Add helper script for adding workers
RUN echo '#!/bin/bash\nif [ -z "$1" ]; then\n    echo "Usage: add-worker <worker-url>"\n    exit 1\nfi\ncurl -X POST "http://${ROUTER_HOST:-localhost}:${ROUTER_PORT:-30000}/add_worker?url=$1"' > /usr/local/bin/add-worker && \
    chmod +x /usr/local/bin/add-worker

# Create an entrypoint script
RUN echo '#!/bin/bash\nif [ "$1" = "router" ]; then\n    shift\n    exec router "$@"\nelif [ "$1" = "server" ]; then\n    shift\n    exec server "$@"\nelif [ "$1" = "add-worker" ]; then\n    shift\n    exec add-worker "$@"\nelse\n    exec "$@"\nfi' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]