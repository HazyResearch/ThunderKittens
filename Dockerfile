FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=`${CUDA_HOME}/bin:`${PATH}
ENV LD_LIBRARY_PATH=`${CUDA_HOME}/lib64:`${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    ninja-build \
    vim \
    nano \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:`${PATH}"

# Accept conda Terms of Service
RUN conda config --set channel_priority flexible && \
    conda tos accept --override-channels --channel defaults && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n thunderkittens python=3.10 -y

SHELL ["conda", "run", "-n", "thunderkittens", "/bin/bash", "-c"]
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    numpy \
    scipy \
    matplotlib \
    jupyter \
    ipython \
    pytest \
    pybind11 \
    setuptools \
    wheel \
    build

WORKDIR /workspace

# Clone the repo but don't install yet
# Clone the repo but don't install yet
RUN git clone https://github.com/HazyResearch/ThunderKittens.git
RUN git config --global http.postBuffer 524288000

# Create proper entrypoint script
RUN printf '#!/bin/bash\n' > /entrypoint.sh && \
    printf 'source /opt/conda/etc/profile.d/conda.sh\n' >> /entrypoint.sh && \
    printf 'conda activate thunderkittens\n' >> /entrypoint.sh && \
    printf 'exec "$@"\n' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]


ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
EXPOSE 8888