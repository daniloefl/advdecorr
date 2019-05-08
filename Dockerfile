FROM python:3

MAINTANER Danilo Ferreira de Lima "daniloefl@gmail.com"

COPY ./app/__init__.py /app/__init__.py
COPY ./app/web.py /app/web.py
COPY ./app/static/ /app/static/
COPY ./app/templates/ /app/templates/
WORKDIR /app

RUN pip install flask flask-bootstrap flask-nav numpy h5py pandas keras matplotlib

ENTRYPOINT [ "python" ]

CMD ["app/web.py"]

