FROM python:3.9.14-slim-buster

LABEL maintainer="Eric Li"

WORKDIR /server

COPY server.py /server

RUN apt update -y \
    && pip install flwr torch torchvision tensorflow \
    && cd /server \
    && touch clients

EXPOSE 8080

VOLUME [ "/server/global_model" ]
VOLUME [ "/server/flwr_logs" ]

CMD ["python3", "/server/server.py"]