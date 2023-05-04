FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update
RUN pip install --upgrade pip
CMD ["bash"]