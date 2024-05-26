# Use an official Python runtime as a base image
FROM python:3.9-slim

# Label the maintainer
LABEL authors="David"

# Set the working directory to /usr/src/app
WORKDIR /usr/src/app

# Copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Command to run on container start
CMD ["python", "./converter.py"]
