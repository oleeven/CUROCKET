FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages sktime numba cupy-cuda12x jupyter