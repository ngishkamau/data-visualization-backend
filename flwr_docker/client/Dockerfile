FROM python:3.9.14-slim-buster

LABEL maintainer="Eric Li"

WORKDIR /client

COPY client.py /client

ARG IP="127.0.0.1"
ARG PORT=8080

RUN apt update -y \
    && pip install flwr torch torchvision tqdm \
    && cd /client \
    && sed -i "s/127.0.0.1:8080/${IP}:${PORT}/g" client.py


VOLUME [ "/data" ]
VOLUME [ "/client/local_model" ]

CMD ["python3", "/client/client.py"]