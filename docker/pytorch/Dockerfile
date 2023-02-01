FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV HOME=/workspace \
    LD_LIBRARY_PATH=/usr/local/lib/ \
    PIP_NO_CACHE_DIR="off"

ARG DEBIAN_FRONTEND="noninteractive"

WORKDIR ${HOME}

EXPOSE 6006 8888

# Add non-root user.
# Name of vscode is arbitrary.
ARG USERNAME="vscode"
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Update GPG key
# https://github.com/NVIDIA/nvidia-docker/issues/1631
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# https://www.nemotos.net/?p=5178
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update \
    && apt-get install -y \
        less \
        git \
        wget \
        curl \
        zsh \
        build-essential \
        ttf-mscorefonts-installer \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY Pipfile Pipfile.lock ${HOME}/

RUN cd ${HOME} \
    && pip install pip==22.2.2 pipenv==2022.8.5 \
    && export PATH="$PATH:${HOME}/.local/bin" \
    && pipenv install --system --dev --ignore-pipfile \
    && rm -rf ${HOME}/.cache/pip ${HOME}/.cache/pipenv

COPY .jupyter/ ${HOME}/.jupyter/

