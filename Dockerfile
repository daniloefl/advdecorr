FROM python:3

ADD production/classify.py production/
ADD keras_gan.py /
ADD keras_aae.py /
ADD utils.py /
ADD generateToys.py /

RUN pip install numpy h5py pandas keras matplotlib

CMD ["python", "./production/classify.py"]

