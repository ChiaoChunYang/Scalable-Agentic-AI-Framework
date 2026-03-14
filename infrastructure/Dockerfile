# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure scripts in .local/bin are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Application environment variables
ENV LOGURU_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Using a generic entrypoint for the agent service
CMD ["python", "src/agents/gtm_agent.py"]
