FROM ubuntu:16.04

RUN apt-get update && \
    apt-get install -y python2.7 python2.7-dev python-pip \
                       build-essential gfortran python-numpy

COPY requirements.txt .
RUN pip install Cython==0.28.5 && pip install -r requirements.txt

COPY . /code
RUN cd /code && python setup.py build_ext --inplace --force
