# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the Shiny app listend on
# Default for Shiny is 8000. Start at 8010
EXPOSE 8009

# Command to run the application
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8009", "app.py"]
