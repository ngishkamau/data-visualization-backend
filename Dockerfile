FROM  tiangolo/uvicorn-gunicorn:python3.9-alpine3.14

RUN apk add mysql mysql-client gcc musl-dev mariadb-connector-c-dev
# py3-setuptools
RUN mkdir -p /app

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt


RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python3" ]

CMD [ "uvicorn", "main:app", "--reload" ]