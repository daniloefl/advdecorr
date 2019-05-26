FROM python:3

LABEL maintainer="daniloefl@gmail.com"

COPY ./keras_gan.py /training/keras_gan.py
COPY ./keras_aae.py /training/keras_aae.py
COPY ./generateToys.py /training/generateToys.py
COPY ./utils.py /training/utils.py
COPY ./test_training.sh /training/test_training.sh
WORKDIR /training

RUN pip install sklearn numpy tables h5py pandas tensorflow keras matplotlib seaborn

ENTRYPOINT [ "./test_training.sh" ]

