FROM ubuntu:18.04

RUN apt-get update
RUN apt install -y git
RUN apt-get install -y python3.7 python3.7-dev python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONIOENCODING=utf-8

CMD ["./start.sh"]