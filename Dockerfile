FROM python:3.8.6-buster


COPY requirements.txt /requirements.txt
COPY Makefile /Makefile
COPY MANIFEST.in /MANIFEST.in
COPY app.py /app.py

COPY api /api

COPY kidney_kids /kidney_kids

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
