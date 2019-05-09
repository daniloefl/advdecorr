FROM python:3

MAINTANER Danilo Ferreira de Lima "daniloefl@gmail.com"

COPY ./classify.py /app/classify.py
WORKDIR /app

RUN pip install flask flask-restful numpy tensorflow h5py keras

EXPOSE 5001

ENTRYPOINT [ "python" ]

CMD ["classify.py"]

