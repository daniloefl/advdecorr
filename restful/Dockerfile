FROM python:3

LABEL maintainer="daniloefl@gmail.com"

COPY ./classify.py /app/classify.py
COPY ./disc.h5     /app/disc.h5
COPY ./disc.json   /app/disc.json

WORKDIR /app

RUN pip install flask flask-restful numpy tensorflow h5py keras

EXPOSE 5001

ENTRYPOINT [ "python" ]

CMD ["classify.py"]

