FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./requirements.txt

RUN python -m pip install --upgrade pip 
RUN pip install -r requirements-deploy.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

CMD uvicorn app.app_api:app --host 0.0.0.0 --port ${PORT}
