FROM python:3

MAINTANER Danilo Ferreira de Lima "daniloefl@gmail.com"

COPY ./__init__.py /app/__init__.py
COPY ./web.py /app/web.py
COPY ./static/ /app/static/
COPY ./templates/ /app/templates/
WORKDIR /app

RUN pip install flask requests flask-bootstrap flask-nav numpy

EXPOSE 80

ENTRYPOINT [ "python" ]

CMD ["web.py"]

