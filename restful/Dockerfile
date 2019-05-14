FROM python:3

LABEL maintainer="daniloefl@gmail.com"

COPY ./app.py      /app/app.py
COPY ./disc.h5     /app/disc.h5
COPY ./disc.json   /app/disc.json
COPY ./api_configuration.yaml /app/api_configuration.yaml

WORKDIR /app

RUN pip install flask connexion[swagger-ui] uwsgi numpy tensorflow h5py keras

EXPOSE 5001

ENTRYPOINT [ "python" ]

CMD ["app.py"]
#CMD ["uwsgi", "--http-socket", ":5001", "-w", "app"]

