FROM python:3

MAINTANER Danilo Ferreira de Lima "daniloefl@gmail.com"

COPY ./app/web.py /app/web.py
WORKDIR /app

RUN pip install Flask numpy h5py pandas keras matplotlib

ENTRYPOINT [ "python" ]

CMD ["app/web.py"]

