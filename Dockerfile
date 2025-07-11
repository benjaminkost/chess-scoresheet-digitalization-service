FROM python:3.12

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./

COPY src ./src

COPY .env .

RUN poetry install

CMD ["poetry", "run", "uvicorn", "src.main:app", "--host=0.0.0.0", "--port=8000"]