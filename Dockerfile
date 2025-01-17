# FROM  tiangolo/uvicorn-gunicorn:python3.9-alpine3.14

# RUN apk add mysql mysql-client gcc musl-dev mariadb-connector-c-dev
# # py3-setuptools
# RUN mkdir -p /app

# WORKDIR /app

# COPY ./requirements.txt /app/requirements.txt


# RUN pip3 install -r requirements.txt

# COPY . /app

# ENTRYPOINT [ "python3" ]

# CMD [ "uvicorn", "main:app", "--reload" ]

FROM  tensorflow/tensorflow:2.9.2
# tiangolo/uvicorn-gunicorn:python3.8-alpine3.10

# python3.9-alpine3.14


RUN apt-get install -y mysql-server mysql-client gcc musl-dev libmariadb3 libmariadb-dev
# mariadb-connector-c-dev
# py3-setuptools
RUN mkdir -p /app

WORKDIR /app


COPY ./requirements.txt /app/requirements.txt


# RUN apt-get -y install default-libmysqlclient-dev
# RUN apt-get install python-dev libfreetype6-dev
RUN pip3 install --upgrade pip setuptools wheel \
    && python3 -m pip install --upgrade pip \
    && pip3 install -r requirements.txt \
    && pip install uvicorn

COPY . /app

# ENTRYPOINT [ "python3" ]

CMD [ "uvicorn", "--host", "0.0.0.0", "--port", "8000" , "main:app"]