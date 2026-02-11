FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    vim \
    neovim \
    wget \
    curl \
    git \
    zsh \
    libxcb-xinerama0 \
    lsb-release \
    gfortran \
    python3-dev \
    python3-pip \
    && apt-get clean all

# install MPICH https://docs.nersc.gov/development/containers/shifter/how-to-use/#using-mpi-in-shifter
ARG mpich=3.4.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch ./configure --with-device=ch4:ofi && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN /sbin/ldconfig

ARG mpi4py=4.0.3
RUN python3 -m pip install mpi4py==$mpi4py

RUN conda init bash

RUN conda install -y jupyter

RUN git clone https://github.com/minyoungg/vqtorch && cd vqtorch && pip install -e .

COPY requirements.txt .
RUN pip install -r requirements.txt


SHELL ["/bin/bash", "--login", "-c"]
