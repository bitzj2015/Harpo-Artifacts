#! /bin/sh
sudo docker image rm openwpm
docker build -f Dockerfile-openwpm -t openwpm:latest --shm-size=8g .