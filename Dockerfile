FROM python:3

MAINTANER Danilo Ferreira de Lima "daniloefl@gmail.com"

#COPY ./app/__init__.py /app/__init__.py
#COPY ./app/web.py /app/web.py
#COPY ./app/static/ /app/static/
#COPY ./app/templates/ /app/templates/
#WORKDIR /app

COPY ./production/classify.py /production/classify.py
WORKDIR /production

#RUN pip install flask flask-restful flask-bootstrap flask-nav numpy h5py pandas keras matplotlib
RUN pip install flask flask-restful numpy h5py pandas keras matplotlib

ENTRYPOINT [ "python" ]

CMD ["production/classify.py"]

