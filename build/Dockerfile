# =============================================================
# Configuration
# =============================================================

ARG UBUNTU_VERSION=20.04
ARG CUDA_MAJOR_VERSION=11.3.1
ARG CUDNN_MAJOR_VERSION=8
ARG PYTHON_VERSION=3.9.15
ARG NUMPY_VERSION=1.21.5
ARG PYTORCH_VERSION=1.10.2
ARG TORCHVISION_VERSION=0.11.3
ARG TENSORFLOW_VERSION=2.8.0 
ARG VIPS_VERSION=8.13.0
ARG PYTHOSTITCHER_VERSION=0.3.1

ARG BUILD_JOBS=16

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
    pip3 install cython wheel numpy==${NUMPY_VERSION} packaging

# =============================================================
# Build Object Detection framework for Tensorflow
# =============================================================

FROM builder AS tensorflow-odf

# === Install protobuf compiler ===
RUN apt-get install -y --no-install-recommends protobuf-compiler

# === Build the library ===
RUN git clone https://github.com/tensorflow/models.git && \
    cd models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python3 setup.py bdist_wheel && \
    mkdir /tensorflow-odf && \
    cp dist/*.whl /tensorflow-odf/

# =============================================================
# Build NVIDIA apex module for PyTorch
# =============================================================

FROM builder AS apex

# === Propagate build args ===
ARG CUDA_MAJOR_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION

# === Install PyTorch ===
RUN CUDA_IDENTIFIER_PYTORCH=`echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g" | cut -c1-5` && \
    pip3 install --no-cache-dir --find-links https://download.pytorch.org/whl/torch_stable.html \
        "torch==${PYTORCH_VERSION}+${CUDA_IDENTIFIER_PYTORCH}" \
        "torchvision==${TORCHVISION_VERSION}+${CUDA_IDENTIFIER_PYTORCH}"

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
		meson \
        curl wget \
        bzip2 zip unzip \
        dcmtk libboost-all-dev \
        libgomp1 \
        libjpeg-turbo8 \
        libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev \
        libgtk2.0-dev libgsf-1-dev libtiff5-dev libopenslide-dev \
        libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd && \
    cat /etc/sudoers | grep -v secure_path > /tmp/sudoers && mv /tmp/sudoers /etc/sudoers

# == Install VIPS  ==
ARG VIPS_VERSION

RUN wget https://github.com/libvips/libvips/releases/download/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.gz -P /tmp && \
    tar -xf /tmp/vips-${VIPS_VERSION}.tar.gz --directory /tmp/ && \
    cd /tmp/vips-${VIPS_VERSION} && \
    ./configure && \
    make && \
    sudo -S make install && \
    cd .. && \
    sudo -S ldconfig

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

# === Install python libraries ===
COPY --from=tensorflow-odf /tensorflow-odf/object_detection-*.whl /root/python-packages/
COPY install_files/requirements.txt /root/python-packages/
RUN pip3 install -r /root/python-packages/requirements.txt

# === Set some environment variables for TensorFlow. ===
ENV FOR_DISABLE_CONSOLE_CTRL_HANDLER 1
ENV TF_CPP_MIN_LOG_LEVEL 3
RUN env | grep '^FOR_DISABLE_CONSOLE_CTRL_HANDLER=\|^TF_CPP_MIN_LOG_LEVEL=' >> /etc/environment

#  === Install ASAP ===
RUN wget https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1/ASAP-2.1-py39-Ubuntu2004.deb -P /root/
RUN apt-get update && \
    apt-get install --assume-yes /root/ASAP-2.1-py39-Ubuntu2004.deb && \
    ldconfig && \
    SITE_PACKAGES=`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    #printf "/opt/ASAP/bin/\n" > "/usr/lib/python3.9/site-packages/asap.pth" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#  === Install latest Pixman ===
#  This should fix the bug described here (https://github.com/TissueImageAnalytics/tiatoolbox/issues/133)
RUN sudo apt-get install meson && \
	wget https://www.cairographics.org/releases/pixman-0.42.2.tar.gz && \
	tar -xf pixman-0.42.2.tar.gz && \
	cd pixman-0.42.2 && \
	mkdir build && \
	cd build && \
	meson --prefix=/usr --buildtype=release && \
	ninja && \
	sudo ninja install

# === Get latest pre-release of PythoStitcher ===
WORKDIR /home/user
ARG PYTHOSTITCHER_VERSION
RUN wget https://github.com/computationalpathologygroup/pythostitcher/archive/refs/tags/${PYTHOSTITCHER_VERSION}.tar.gz && \
    tar xf ${PYTHOSTITCHER_VERSION}.tar.gz

# === Download JigsawNet model weights in PythoStitcher directory ===
# The weights will be downloaded from my Google Drive since they were modified to fit PythoStitcher. 
# The original JigsawNet weights (https://github.com/Lecanyu/JigsawNet) will NOT work. 
ENV FILE_ID 1jmee9j5dKemu62MNl7UwXx8_0QrXYwJU
RUN wget --save-cookies cookies.txt 'https://drive.google.com/uc?export=download&id='$FILE_ID -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
RUN wget --load-cookies cookies.txt -O JigsawCNN_checkpoint.zip \
     'https://drive.google.com/uc?export=download&id='$FILE_ID'&confirm='$(<confirm.txt)
RUN unzip /home/user/JigsawCNN_checkpoint.zip -d /home/user/pythostitcher-${PYTHOSTITCHER_VERSION}/model_weights/

# === Configure stop signal. ===
STOPSIGNAL SIGINT

# === Expose ports for ssh, tensorboard, and jupyter. ===
EXPOSE 22 6006 8888

# === Overwrite NVIDIA's authorship label
LABEL maintainer="DNSCHOUTEN"

# === Set entrypoint to run PythoStitcher ===
USER root
WORKDIR /home/user/pythostitcher-${PYTHOSTITCHER_VERSION}/src
RUN cd /home/user/pythostitcher-${PYTHOSTITCHER_VERSION}/src

COPY install_files/run.sh /root/run.sh
ENTRYPOINT ["/bin/bash", "/root/run.sh"]
