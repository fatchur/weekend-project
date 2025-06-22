FROM python:3.9-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY my_rest_api.py .

EXPOSE 10000

ENV FLASK_APP=my_rest_api.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["python", "my_rest_api.py"]