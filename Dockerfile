# Minoconda 3 dockerfile, base image is changed
# ubuntu:latest
FROM nvidia/cuda:10.0-cudnn7-devel

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    # IMPORTANT!: Create symbolic link -> This way 'conda activate _env_' will work
    ln -snf /bin/bash /bin/sh


### Set timezone & stuff for python datetime
ENV TZ=Europe/Budapest
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
### Create conda env ###
COPY generali_lpr_env.yml /root
RUN conda env create -f /root/generali_lpr_env.yml && \
    echo "conda activate generali_demo" >> ~/.bashrc && \
    # Delete cache and other stuffz
    conda clean -a -y
ENV PATH /opt/conda/envs/generali_demo/bin:$PATH

WORKDIR /alpr-unconstrained

COPY . .

WORKDIR /alpr-unconstrained/darknet

RUN sed -i 's/GPU=0/GPU=1/g' Makefile

RUN make

# ENV PYTHONPATH='/src/:$PYTHONPATH'
WORKDIR /alpr-unconstrained

RUN mkdir -p /mnt

EXPOSE 5000

CMD [  "python", "service.py" ]