FROM python:3.10-slim

# 1) Install git
RUN apt-get update \
 && apt-get install -y git \
 && rm -rf /var/lib/apt/lists/*

# 2) Add a non-root user
RUN useradd -m -s /bin/bash user \
 && chown -R user /home/user

# 3) Copy the *entire* comps/ package
WORKDIR /home/user
COPY comps /home/user/comps

# 4) Install deps from the router’s requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /home/user/comps/router/src/requirements.txt

# 5) Make imports work
ENV PYTHONPATH=/home/user

# 6) Switch to non-root
USER user

# 7) Expose the port
EXPOSE 6000

# 8) Run the microservice
WORKDIR /home/user/comps/router/src
CMD ["python", "opea_router_microservice.py"]
