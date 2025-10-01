# 基底映像: Ubuntu 20.04 + CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 設定非互動模式
ENV DEBIAN_FRONTEND=noninteractive

# 更新套件並安裝必要工具
RUN apt-get update && apt-get install -y screen tree sudo ssh synaptic psmisc aptitude gedit geany \
    wget \
    curl \
    gnupg2 \
    mesa-utils \
    lsb-release \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
    


# 安裝 ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update \
    && apt-get install -y ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y \
    python3-pip \
    python-is-python3 \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    && rm -rf /var/lib/apt/lists/*
    
# 安裝 catkin tools
RUN pip3 install -U catkin_tools

# ROS deps
RUN apt-get update && apt-get install -y \
    doxygen \
    libssh2-1-dev \
    libudev-dev \
    ros-noetic-turtlebot3-msgs \
    ros-noetic-vision-msgs \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-move-base-msgs \
    ros-noetic-costmap-converter \
    ros-noetic-mbf-costmap-core \
    ros-noetic-mbf-msgs \
    libsuitesparse-dev \
    ros-noetic-libg2o \
    ros-noetic-navigation \
    ros-noetic-turtlebot3-description \
    && rm -rf /var/lib/apt/lists/*
    

# 初始化 rosdep
RUN rosdep init || true && rosdep update

# 設定環境變數
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 建立工作區
WORKDIR /workspace
RUN mkdir -p /workspace/src

# 安裝 Python 套件 (包含 YOLO 可能需要的 PyTorch)
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install numpy opencv-python
RUN pip3 install ultralytics
RUN pip3 install filterpy
RUN pip3 install libsvm

# 預設啟動 bash
CMD ["/bin/bash"]

