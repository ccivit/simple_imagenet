FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get install --allow-downgrades --allow-remove-essential --allow-change-held-packages -y libcudnn7=7.0.5.15-1+cuda9.0
RUN apt-get install -y python3-dev python3-pip python3-nose python3-numpy python3-scipy
RUN pip3 install --upgrade pip tensorflow-gpu keras Pillow PyYAML

WORKDIR /keras

ENTRYPOINT ["python3", "imagenet.py"]
