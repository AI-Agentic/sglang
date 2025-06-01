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
RUN git clone -b jinleic-docker https://github.com/AI-Agentic/sglang

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
# RUN rm /tmp/pyarmor-license.zip

# 1. Obfuscate
RUN pyarmor gen -O /tmp/obf -r python/sglang

# 2. Copy the obfuscated files over the originals
RUN cp -a /tmp/obf/sglang/. python/sglang/


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
RUN pip install sglang-router

CMD ["bash"]