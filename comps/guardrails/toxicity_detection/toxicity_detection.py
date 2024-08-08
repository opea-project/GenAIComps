# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langsmith import traceable

# from utils import initialize_model
from transformers import pipeline

from comps import TextDoc, ServiceType, opea_microservices, register_microservice


@register_microservice(
    name="opea_service@toxicity_detection",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/toxicity",
    host="0.0.0.0",
    port=9091,
    input_datatype=TextDoc,
    output_datatype=TextDoc,
)
@traceable(run_type="llm")
def llm_generate(input: TextDoc):
    input_text = input.text
    toxic = toxicity_pipeline(input_text)
    print('done')
    if toxic[0]["label"] == "toxic":
        return TextDoc(text=f"Violated policies: toxicity, please check your input.", downstream_black_list=[".*"])
    else:
        return TextDoc(text=input_text)


if __name__ == "__main__":
    model = "citizenlab/distilbert-base-multilingual-cased-toxicity"
    toxicity_pipeline = pipeline("text-classification", model=model, tokenizer=model)
    opea_microservices["opea_service@toxicity_detection"].start()
