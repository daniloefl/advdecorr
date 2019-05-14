FROM python:3

LABEL maintainer="daniloefl@gmail.com"

COPY ./__init__.py /app/__init__.py
COPY ./web.py /app/web.py
COPY ./static/ /app/static/
COPY ./templates/ /app/templates/
WORKDIR /app

RUN pip install flask requests flask-bootstrap flask-nav

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD ["web.py"]

