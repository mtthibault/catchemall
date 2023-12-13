# Base layer
# for scikit learn
# FROM python:3.10-slim-buster
FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

# First, pip install dependencies
# ---------------------------------------------------------
# COPY requirements.txt requirements.txt # local
COPY cnn/cnn_requirements.txt requirements.txt

# Installing the requirements, dependencies of our packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Application set-up
# ---------------------------------------------------------
COPY cnn cnn
# COPY raw_data raw_data
COPY raw_data/computer_vision/pokemon.csv raw_data/computer_vision/pokemon.csv
# COPY raw_data/computer_vision raw_data/computer_vision
# COPY

# Double-check
RUN find cnn
RUN find raw_data

# Install app
COPY setup.py setup.py
RUN pip install .

# Testing
# COPY Makefile Makefile

# Web app settings, 8080 for GCP
# ---------------------------------------------------------
CMD uvicorn cnn.api.fast:app --host 0.0.0.0 --port $PORT

# for GCP
# CMD uvicorn cnn.api.fast:app --host 0.0.0.0 --port $PORT

# local DEV version
# CMD uvicorn cnn.api.fast:app --host 0.0.0.0
