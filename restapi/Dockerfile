FROM python:3.9-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pkg ./pkg

EXPOSE 10000

ENV FLASK_APP=pkg.controller.statistic
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "pkg.controller.statistic"]