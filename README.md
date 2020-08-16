# Simple ImageNet classifier
Minimalist implementation of ImageNet prediction using ResNet101 with Keras and Docker

```
git clone https://github.com/ccivit/simple_imagenet
cd simple_imagenet
sudo nvidia-docker build -t imagenet .
```
This will build the docker. In order to get the imagenet classification for a specific image, run the following command: 

```
nvidia-docker run -it -v /volume/to/map:/keras imagenet /image/path
```
