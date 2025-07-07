# Dockerfile.rocm
FROM rocm/dev-ubuntu-20.04:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libclang-dev \
    curl

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

WORKDIR /app
COPY . .

# Build with ROCm support
RUN cargo build --features rocm
