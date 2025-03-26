FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy application code
COPY src/ ./src/

# Default port for SSE
ENV PORT=8080
# Default to stdio transport
ENV USE_SSE=false

# Expose port for SSE
EXPOSE 8080

CMD ["python", "src/main.py"] 