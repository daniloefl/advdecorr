FROM python:3

ADD production/classify.py production/
ADD keras_gan.py /
ADD keras_aae.py /
ADD generateToys.py

