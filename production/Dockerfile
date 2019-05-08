FROM python:3

MAINTANER Danilo Ferreira de Lima "daniloefl@gmail.com"

COPY ./classify.py /classify.py
WORKDIR /

RUN pip install flask flask-restful numpy h5py pandas keras matplotlib

ENTRYPOINT [ "python" ]

CMD ["classify.py"]

