FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY dep_requirements.txt .

RUN pip install --no-cache-dir -r dep_requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]