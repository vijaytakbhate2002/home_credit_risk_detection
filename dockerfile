FROM python:3.13-slim

WORKDIR /app

COPY dep_requirements.txt .

RUN pip install --no-cache-dir -r dep_requirements.txt

COPY . .

EXPOSE 8000

CMD ["python","app.py","--host","0.0.0.0","--port","8000"]