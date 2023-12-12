# Base layer
# for scikit learn
FROM python:3.10-slim-buster
# FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

# First, pip install dependencies
# ---------------------------------------------------------
# COPY requirements.txt requirements.txt # local
COPY prediction/pr_requirements.txt requirements.txt

# Installing the requirements, dependencies of our packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Application set-up
# ---------------------------------------------------------
COPY prediction prediction
COPY raw_data raw_data
# COPY

# Double-check
RUN find prediction
RUN find raw_data

# Install app
COPY setup.py setup.py
RUN pip install .

# Testing
COPY Makefile Makefile
# A tester avant
# RUN make reset_local_files

# Web app settings, 8080 for GCP
# ---------------------------------------------------------
CMD uvicorn prediction.api.fast:app --host 0.0.0.0 --port $PORT

# for GCP
# CMD uvicorn prediction.api.fast:app --host 0.0.0.0 --port $PORT

# local DEV version
# CMD uvicorn prediction.api.fast:app --host 0.0.0.0
