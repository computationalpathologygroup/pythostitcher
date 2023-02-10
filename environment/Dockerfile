# =============================================================
# Configuration
# =============================================================

ARG UBUNTU_VERSION=20.04
ARG CUDA_MAJOR_VERSION=11.3.1
ARG CUDNN_MAJOR_VERSION=8
ARG PYTHON_VERSION=3.8.10
ARG NUMPY_VERSION=1.22.3
ARG PYTORCH_VERSION=1.11.0
ARG TORCHVISION_VERSION=0.12.0
ARG TENSORFLOW_VERSION=2.9.0

ARG BUILD_JOBS=16

ARG CODE_DIR="/home/user/source"

# =============================================================
# Create build docker
# =============================================================

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# === Propagate build args ===
ARG PYTHON_VERSION
ARG NUMPY_VERSION
ARG BUILD_JOBS

# === Install build packages ===
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        pkg-config apt-transport-https \
        openjdk-8-jdk \
        g++ ninja-build make \
        wget git zip unzip \
        libssl-dev zlib1g-dev \
        libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
        gcovr libffi-dev uuid-dev

# === Install python ===
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd /tmp && \
    rm -rf Python-3* && \
    ldconfig

RUN pip3 install --upgrade pip && \
    pip3 install cython wheel


# =============================================================
# Build Object Detection framework for Tensorflow
# =============================================================

#FROM builder AS tensorflow-odf

# === Install protobuf compiler ===
#RUN apt-get install -y --no-install-recommends protobuf-compiler

# === Build the library ===
#RUN git clone https://github.com/tensorflow/models.git && \
#    cd models/research && \
#    protoc object_detection/protos/*.proto --python_out=. && \
#    cp object_detection/packages/tf2/setup.py . && \
#    python3 setup.py bdist_wheel && \
#    mkdir /tensorflow-odf && \
#    cp dist/*.whl /tensorflow-odf/


# =============================================================
# Build NVIDIA apex module for PyTorch
# =============================================================

FROM builder AS apex

# === Propagate build args ===
ARG CUDA_MAJOR_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION

# === Install PyTorch ===
#RUN CUDA_IDENTIFIER_PYTORCH=`echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g" | cut -c1-5` && \
#    pip3 install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
#        "torch==${PYTORCH_VERSION}+${CUDA_IDENTIFIER_PYTORCH}" \
#        "torchvision==${TORCHVISION_VERSION}+${CUDA_IDENTIFIER_PYTORCH}"

# === Build the library ===
#ENV TORCH_CUDA_ARCH_LIST "5.0;6.0;6.1+PTX;7.0;7.5;8.0"
#RUN git clone https://github.com/NVIDIA/apex /tmp/apex && \
#    cd /tmp/apex && \
#    python3 setup.py --cpp_ext --cuda_ext bdist_wheel && \
#    mkdir /apex && \
#    cp dist/*.whl /apex/


# =============================================================
# Create base docker
# =============================================================

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

# === Propagate build args ===
ARG CUDA_MAJOR_VERSION
ARG PYTHON_VERSION
ARG NUMPY_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TENSORFLOW_VERSION
ARG BUILD_JOBS

# === Configurate environment variables ===
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# === Install libs ===
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        man \
        openssh-server \
        ca-certificates apt-transport-https \
        sudo \
        git subversion \
        nano vim \
        tmux screen \
        htop \
        g++ ninja-build \
        rsync \
        pv \
        curl wget \
        bzip2 zip unzip \
        dcmtk libboost-all-dev \
        libgomp1 \
        libjpeg-turbo8 \
        libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev \
        libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd && \
    cat /etc/sudoers | grep -v secure_path > /tmp/sudoers && mv /tmp/sudoers /etc/sudoers

RUN env | grep '^PATH=\|^LD_LIBRARY_PATH=\|^LANG=\|^LC_ALL=\|^CUDA_ROOT=' > /etc/environment

# === Configure timezone ===
RUN echo "Europe/Amsterdam" > /etc/timezone && \
    rm -f /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# === Setup user ===
RUN useradd -ms /bin/bash user && \
    (echo user ; echo user) | passwd user && \
    gpasswd -a user ssh && \
    gpasswd -a user sudo

# === Install python with up-to-date pip ===
RUN cd /tmp && \
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd ~ && \
    rm -rf /tmp/Python-3* && \
    ldconfig

RUN pip3 install --upgrade pip pip-tools wheel setuptools && \
    printf '#!/bin/bash\necho "Please use pip3 instead of pip to install packages for python3"' > /usr/local/bin/pip && \
    chmod +x /usr/local/bin/pip && \
    rm -rf ~/.cache/pip

#  === Install ASAP ===
COPY install_files/ASAP*.deb /root/
RUN apt-get update && \
    apt-get install --assume-yes /root/ASAP-2.1-py38-Ubuntu2004.deb && \
    ldconfig && \
    SITE_PACKAGES=`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#  === Install libvips ===
WORKDIR /home/user/
RUN wget -q https://github.com/libvips/libvips/releases/download/v8.13.0/vips-8.13.0.tar.gz  && \
    sudo -S apt-get update && \
    sudo -S apt-get install -y libjpeg-turbo8-dev && \
    sudo -S apt-get install -y libgtk2.0-dev && \
    sudo -S apt-get install -y libgsf-1-dev && \
    sudo -S apt-get install -y libtiff5-dev && \
    sudo -S apt-get install -y libopenslide-dev && \
    tar xf vips-8.13.0.tar.gz > /dev/null 2>&1

WORKDIR /home/user/vips-8.13.0/

RUN ./configure && \
    make && \
    sudo -S make install && \
    cd .. && \
    sudo -S ldconfig && \
    sudo -S pip3 install pyvips

# === Install python libraries ===
COPY install_files/requirements.in /root/python-packages/
#COPY --from=tensorflow-odf /tensorflow-odf/object_detection-*.whl /root/python-packages/
#COPY --from=apex /apex/apex-*-linux_x86_64.whl /root/python-packages/

RUN cd /root/python-packages && \
    CUDA_IDENTIFIER_PYTORCH=`echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g" | cut -c1-5` && \
    sed -i \
        -e "s|%NUMPY_VERSION%|${NUMPY_VERSION}|g" \
        -e "s|%TENSORFLOW_VERSION%|${TENSORFLOW_VERSION}|g" \
        -e "s|%PYTORCH_VERSION%|${PYTORCH_VERSION}+${CUDA_IDENTIFIER_PYTORCH}|g" \
        -e "s|%TORCHVISION_VERSION%|${TORCHVISION_VERSION}+${CUDA_IDENTIFIER_PYTORCH}|g" \
        requirements.in && \
    pip-compile requirements.in --verbose --find-links https://download.pytorch.org/whl/torch_stable.html && \
    pip-sync && \
    rm -rf ~/.cache/pip*

#        -e "s|%TENSORFLOW_ODF_WHEEL%|`find /root/python-packages -name 'object_detection-*.whl'`|g" \
#        -e "s|%APEX_WHEEL%|`find /root/python-packages -name 'apex-*.whl'`|g" \


# === Set some environment variables for TensorFlow. ===
ENV FOR_DISABLE_CONSOLE_CTRL_HANDLER 1
ENV TF_CPP_MIN_LOG_LEVEL 3
RUN env | grep '^FOR_DISABLE_CONSOLE_CTRL_HANDLER=\|^TF_CPP_MIN_LOG_LEVEL=' >> /etc/environment

# === Configure stop signal. ===
STOPSIGNAL SIGINT

# === Expose ports for ssh, tensorboard, and jupyter. ===
# EXPOSE 22 6006 8888

# === Provide test scripts ===
#COPY test_files/*.py /opt/tests/base_image/
#RUN wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/builders/model_builder_tf2_test.py \
#         -O /opt/tests/base_image/test_tensorflow_odf.py

# === Overwrite NVIDIA's authorship label
LABEL maintainer="DIAG"

# === Set entrypoint. ===
WORKDIR /home/user
USER root
COPY run_files/run.sh /root/run.sh
ENTRYPOINT ["/bin/bash", "/root/run.sh"]
