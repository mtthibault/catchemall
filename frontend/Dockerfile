FROM python:3.10-slim-buster

WORKDIR /app

# First, pip install dependencies
# ---------------------------------------------------------
# COPY requirements.txt requirements.txt # local
COPY requirements.txt requirements.txt

# Installing the requirements, dependencies of our packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Application set-up
# ---------------------------------------------------------
COPY images images
COPY pages pages
COPY Home.py Home.py
COPY MANIFEST.in MANIFEST.in
# COPY

# Install app
COPY setup.py setup.py
RUN pip install .

# Testing if required
COPY Makefile Makefile

# Web app settings, 8080 for GCP
# ---------------------------------------------------------
CMD streamlit run Home.py --server.port 8080 --server.address=0.0.0.0
