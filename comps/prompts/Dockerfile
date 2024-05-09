FROM python:3.9.18-alpine3.19

WORKDIR /app

COPY ./app /app
COPY ./requirements.txt /app/

RUN apk update && apk upgrade && apk add expat=2.6.2-r0
RUN pip install setuptools --upgrade
RUN pip install pip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5433
USER 9000:9000

ENTRYPOINT ["python3","main.py"]
