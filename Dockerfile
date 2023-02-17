ARG UBUNTU_VERSION=18.04
ARG CUDA_VERSION=11.6.0
ARG PYTORCH_CUDA=11.6
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

ARG PYTHON_VERSION=3.8
# To use the default value of an ARG declared before the first FROM,
# use an ARG instruction without a value inside of a build stage:
ARG CUDA_VERSION

# Install ubuntu packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    locales \
    python3-opencv \
    libgtk2.0-dev \
    libjpeg-dev \
    libpng-dev \
    byobu \
    htop \
    vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    # Make the "en_US.UTF-8" locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Setup timezone
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install miniconda (python)
# Referenced PyTorch's Dockerfile:
#   https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p conda && \
    rm miniconda.sh && \
    conda/bin/conda install -y python=$PYTHON_VERSION jupyter jupyterlab && \
    conda/bin/conda install -y pytorch torchvision torchaudio pytorch-cuda=$PYTORCH_CUDA -c pytorch -c nvidia && \
    conda/bin/conda clean -ya
ENV PATH $HOME/conda/bin:$PATH


RUN pip install --upgrade pip
RUN pip install flake8 typing mypy pytest pytest-mock
RUN pip install ufmt==2.0.0 black==22.6.0 usort==1.0.4

ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace/pytv
RUN python setup.py develop
