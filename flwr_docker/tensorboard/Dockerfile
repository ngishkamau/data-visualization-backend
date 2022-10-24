FROM python:3.9.14-slim-buster

ENV maintainer="Eric Li"

RUN apt update -y \
    && mkdir /logs \
    && pip install tensorboard

EXPOSE 8090

VOLUME [ "/logs" ]

CMD ["tensorboard", "--logdir", "/logs", "--host", "0.0.0.0", "--port", "8090"]