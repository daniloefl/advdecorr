FROM python:3

MAINTANER Danilo Ferreira de Lima "daniloefl@gmail.com"

COPY ./__init__.py /__init__.py
COPY ./web.py /web.py
COPY ./static/ /static/
COPY ./templates/ /templates/
WORKDIR /

RUN pip install flask flask-restful flask-bootstrap flask-nav numpy h5py pandas keras matplotlib

ENTRYPOINT [ "python" ]

CMD ["web.py"]

