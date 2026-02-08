FROM ubuntu:24.04
RUN apt update -y && apt upgrade -y \
	&& apt install -y libgl1-mesa-dev libglib2.0-0 python3-pip
RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED
RUN apt install --only-upgrade python3-pip \
	&& pip install retina-face opencv-python mct-quantizers tensorflow
WORKDIR /usr/src/app