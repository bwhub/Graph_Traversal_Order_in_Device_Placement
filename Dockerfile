# FROM tensorflow/tensorflow:1.9.0-gpu-py3
FROM tensorflow/tensorflow:1.9.0-py3

RUN mkdir -p placeto

WORKDIR /placeto

RUN mkdir -p datasets

RUN apt update
RUN apt install -y tmux nano htop
RUN apt install -y python-pygraphviz graphviz libgraphviz-dev

ADD ./requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
