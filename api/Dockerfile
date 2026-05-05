FROM python:3.11-slim

WORKDIR /app

# Copy the entire requirements directory
COPY requirements/ /app/requirements/

# Install ONLY the API dependencies
RUN pip install --no-cache-dir -r requirements/api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
