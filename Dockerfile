FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# OS dependecies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888
EXPOSE 8501

# default command
CMD ["/bin/bash"]