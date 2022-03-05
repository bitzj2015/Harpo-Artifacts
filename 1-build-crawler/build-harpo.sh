#! /bin/sh
sudo docker image rm openwpm-harpo
docker build -f Dockerfile-harpo -t openwpm-harpo:latest  --shm-size=8g  .
