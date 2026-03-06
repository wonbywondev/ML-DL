FROM python:3.12-slim

WORKDIR /notebooks

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY notebooks/main.py notebooks/main.py
COPY data/raw data/raw
RUN mkdir -p /data/model

CMD ["python", "main.py"]