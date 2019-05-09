FROM python:3

MAINTANER Danilo Ferreira de Lima "daniloefl@gmail.com"

COPY ./keras_gan.py /app/keras_gan.py
COPY ./keras_aae.py /app/keras_aae.py
COPY ./generateToys.py /app/generateToys.py
WORKDIR /app

RUN pip install numpy h5py pandas tensorflow keras matplotlib

ENTRYPOINT [ "python" ]

CMD ["app/keras_gan.py"]

