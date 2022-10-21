FROM python:3.9.14-slim-buster

LABEL maintainer="Eric Li"

WORKDIR /server

COPY server.py /server

RUN apt update -y \
    && pip install flwr torch torchvision tqdm \
    && cd /server

EXPOSE 8080

VOLUME [ "/server/clients" ]

CMD ["python3", "/server/server.py"]