FROM python:3

LABEL maintainer="daniloefl@gmail.com"

COPY ./keras_gan.py /training/keras_gan.py
COPY ./keras_aae.py /training/keras_aae.py
COPY ./generateToys.py /training/generateToys.py
WORKDIR /training

RUN pip install numpy h5py pandas tensorflow keras matplotlib

ENTRYPOINT [ "python" ]

CMD ["training/generateToys.py"]
CMD ["training/keras_gan.py"]

