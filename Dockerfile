FROM ubuntu:24.04
RUN apt update -y && apt upgrade -y \
	&& apt install -y libgl1-mesa-dev libglib2.0-0 pip
RUN python3 -m pip install --upgrade pip \
	&& python3 -m pip install --break-system-packages retina-face opencv-python mct-quantizers tensorflow
WORKDIR /usr/src/app