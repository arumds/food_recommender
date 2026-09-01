FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/app.py src/data_loader.py src/ranking_fe.py src/constraints.py src/



EXPOSE 8000

CMD ["python3", "src/app.py"]
