FROM python:3.7.10-slim-buster

WORKDIR /app

COPY ./requirements.txt /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

COPY . /app

ENV SERVICE_HOST="0.0.0.0"
EXPOSE 8000
ENTRYPOINT ["python", "main.py"]