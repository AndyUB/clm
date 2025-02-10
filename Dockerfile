FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
RUN pip install torch requests pandas transformers
