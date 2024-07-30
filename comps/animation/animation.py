import requests
import pathlib
import os

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../")

from comps import ServiceType, opea_microservices, register_microservice
from comps import Base64ByteStrDoc
from src.utils import *

args = get_args()
print("args: ", args)

# Register the microservice
@register_microservice(
    name="opea_service@animation",
    service_type=ServiceType.ANIMATION,
    endpoint="/v1/animation",
    host="0.0.0.0",
    port=7860,
    input_datatype=Base64ByteStrDoc,
)
def animate(input: Base64ByteStrDoc):
    print("args: ", args)
    return


if __name__ == "__main__":
    opea_microservices["opea_service@animation"].start()


