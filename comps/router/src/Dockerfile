FROM python:3.10-slim

# Install git (for pip installs from Git if needed)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Create a user (optional – you can follow the same approach as retrievers)
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

WORKDIR /home/user

# Copy the entire 'comps/' folder from your build context. 
COPY comps /home/user/comps

# If you have a separate 'requirements.txt' just for the router:
COPY comps/router/src/requirements.txt /home/user/comps/router/src/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /home/user/comps/router/src/requirements.txt

# Add /home/user to PYTHONPATH so 'comps' is discoverable
ENV PYTHONPATH=${PYTHONPATH}:/home/user

# Switch to normal user (optional, for security)
USER user

# Expose the port
EXPOSE 6000

# Switch to the microservice directory
WORKDIR /home/user/comps/router/src

# Entry point that starts your router microservice
CMD ["python", "opea_router_microservice.py"]
