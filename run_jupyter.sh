#!/usr/bin/env bash

docker run -it -p 8888:8888 \
	-v "${PWD}":/home/jovyan/work quay.io/jupyter/pytorch-notebook \
	start-notebook.py --IdentityProvider.token=''

