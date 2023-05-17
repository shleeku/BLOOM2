FROM python:3.9

LABEL maintainer="VIP701"

RUN mkdir /project
WORKDIR /project
COPY . /project

RUN pip install -r requirements.txt
ENTRYPOINT ["python", "main.py"]