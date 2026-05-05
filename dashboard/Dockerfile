# dashboard/Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies if needed (e.g., for psycopg2)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy the entire requirements directory
COPY requirements/ /app/requirements/

# Install ONLY the Dashboard dependencies
RUN pip install --no-cache-dir -r requirements/dashboard.txt

# Copy the dashboard code
COPY dashboard/ /app/dashboard/

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]