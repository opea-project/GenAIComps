import requests
import pathlib
import os

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../")

from comps import ServiceType, opea_microservices, register_microservice

# Register the microservice
