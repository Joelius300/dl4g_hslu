# syntax=docker/dockerfile:1.4
FROM python:3.11-alpine AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . /app
#COPY ./requirements_common.txt /app
#COPY ./requirements_prod.txt /app
#COPY ./jass_kit/jass/ /app/jass_kit
#COPY ./jass_kit/README.md /app/jass_kit
#COPY ./jass_kit/setup.py /app/jass_kit

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements_prod.txt
