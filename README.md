# Harpo-Artifacts (Ongoing)
## Overview
Harpo is a principled learning-based approach to subvert online behavioral advertising. It uses reinforcement learning to adaptively interleave real page visits with fake pages to distort a tracker’s view of a user’s browsing profile.

## Accepted paper
Jiang Zhang, Konstantinos Psounis, Muhammad Haroon, Zubair Shafiq. [HARPO: Learning to Subvert Online Behavioral Advertising](https://arxiv.org/abs/2111.05792) [C]. NDSS, 2022.

## Crawler
1. The `build-crawler` directory contains our scirpt for building our crawling infrastructure. 
You can firstly run `./build-openwpm.sh` to build a docker image for OpenWPM (https://github.com/openwpm/OpenWPM/tree/firefox-52).
Then you can run `./build-harpo.sh` to build a docker image for running crawling experiments in our Harpo paper.
The main function we use for crawling is `run_ml_crawl.py`, which defines the workflow of crawling experiments. Note that `config.json` defines the configuration of our browser used during experiments.
2. The `run-crawler` directory contains our script for running crawling experiments. The main class we use is called `docker_api` in `docer_api.py`. It will launch a docker container with `openwpm-harpo` image and run command `sudo /usr/bin/python3 /opt/OpenWPM/run_ml_crawl.py --cfg '{}'` in the container.