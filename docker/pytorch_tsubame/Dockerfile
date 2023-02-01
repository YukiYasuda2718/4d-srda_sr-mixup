FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

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

RUN apt-get update \
    && apt-get install -y \
        less \
        git \
        wget \
        curl \
        zsh \
        build-essential \
        python3.8 \
        python3-pip \
        ttf-mscorefonts-installer \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Change python version to 3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

COPY Pipfile Pipfile.lock ${HOME}/

RUN cd ${HOME} \
    && pip3 install pip==22.2.2 pipenv==2022.8.5 \
    && export PATH="$PATH:${HOME}/.local/bin" \
    && pipenv install --system --dev --ignore-pipfile \
    && rm -rf ${HOME}/.cache/pip ${HOME}/.cache/pipenv
