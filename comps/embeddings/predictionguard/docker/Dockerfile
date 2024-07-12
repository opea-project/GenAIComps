# Use a base image with Python installed
FROM python:3.9

# Create a user and set up the environment
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

USER user

# Set the working directory
WORKDIR /home/user

# Copy the relevant directories and files into the container
COPY ../../ /home/user/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /home/user/embeddings/predictionguard/requirements.txt

# Set environment variable for the API key
ENV PREDICTIONGUARD_API_KEY=${PREDICTIONGUARD_API_KEY}

# Expose the port the app runs on
EXPOSE 6000

# Run the service
CMD ["python", "/home/user/embeddings/predictionguard/pg_embedding.py"]