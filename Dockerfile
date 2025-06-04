FROM tensorflow/tensorflow:2.15.0-gpu

RUN mkdir /usenix2025

WORKDIR /usenix2025

COPY requirements.txt /usenix2025

RUN pip install --no-cache-dir -r requirements.txt

COPY src/*.py /usenix2025
COPY src/*.sh /usenix2025
COPY src/datasets /usenix2025/datasets