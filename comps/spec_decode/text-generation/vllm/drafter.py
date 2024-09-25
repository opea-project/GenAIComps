# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import base64
import os
import pickle
import threading
from typing import Dict, Tuple

from vllm import LLM

from comps import CustomLogger, ServiceType, SpecDecodeDoc, opea_microservices, opea_telemetry, register_microservice

logger = CustomLogger("spec_decode_drafter_vllm")
logflag = os.getenv("LOGFLAG", False)

llm_instances: Dict[Tuple[str, str, int], LLM] = {}
lock = threading.Lock()

def get_llm(spec_step, device, tensor_parallel_size): 
    model_name = os.getenv("LLM_MODEL", "facebook/opt-6.7b")
    speculative_model = os.getenv("SPEC_MODEL", "facebook/opt-125m")
    num_speculative_tokens = os.getenv("NUM_SPECULATIVE_TOKENS", 5)
    global llm_instances
    with lock:
        if (model_name, speculative_model, num_speculative_tokens) not in llm_instances:
            llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                speculative_model=speculative_model,
                num_speculative_tokens=num_speculative_tokens,
                use_v2_block_manager=True,
                spec_step=spec_step,
                device=device,
            )
            llm_instances[(model_name, speculative_model, num_speculative_tokens)] = llm
        return llm_instances[(model_name, speculative_model, num_speculative_tokens)]


# Create an Drafter.
@register_microservice(
    name="opea_service@spec_decode_draft_vllm",
    service_type=ServiceType.SPEC_DECODE_DRAFTER,
    endpoint="/v1/spec_decode/draft/completions",
    host="0.0.0.0",
    port=8016,
)
def llm_generate(input: SpecDecodeDoc):
    if logflag:
        logger.info(input)
    # reuse the llm model
    llm = get_llm(spec_step='draft', device='cuda', tensor_parallel_size=1)
    execute_model_req = pickle.loads(base64.b64decode(input.model_req))
    assert (
        execute_model_req.seq_group_metadata_list is not None
    ), "speculative decoding requires non-None seq_group_metadata_list"
    print("execute model req post right...")

    # import pdb; pdb.set_trace()
    if input.no_spec:
        sampler_output = llm.llm_engine.model_executor.driver_worker.execute_model(execute_model_req)
    else:
        sampler_output = llm.llm_engine.model_executor.driver_worker.get_spec_proposals(
            execute_model_req, input.seq_with_bonus_token_in_last_step
        )
    print("sampler_output done with {}".format(sampler_output))

    serialized_output = pickle.dumps(sampler_output)
    base64_output = base64.b64encode(serialized_output).decode("utf-8")

    return base64_output


if __name__ == "__main__":
    # TODO This is for test the functionality
    # model_req = 'gASVpwUAAAAAAACMDXZsbG0uc2VxdWVuY2WUjBNFeGVjdXRlTW9kZWxSZXF1ZXN0lJOUKF2UKGgAjBVTZXF1ZW5jZUdyb3VwTWV0YWRhdGGUk5QojAEwlIh9lEsAaACMDFNlcXVlbmNlRGF0YZSTlCiMBWFycmF5lIwUX2FycmF5X3JlY29uc3RydWN0b3KUk5QojAVhcnJheZSMBWFycmF5lJOUjAFslEsMQzACAAAAAAAAALZ6AAAAAAAABgAAAAAAAAB/AAAAAAAAAP4CAAAAAAAAEAAAAAAAAACUdJRSlGgMKGgPaBBLDEMAlHSUUpRHAAAAAAAAAAAoSwJNtnpLBkt/Tf4CSxB0lEsAaACMDVNlcXVlbmNlU3RhZ2WUk5RLAYWUUpRdlChLAk22eksGS39N/gJLEGVdlHSUUpRzjBR2bGxtLnNhbXBsaW5nX3BhcmFtc5SMDlNhbXBsaW5nUGFyYW1zlJOUKEsBSwFHAAAAAAAAAABHAAAAAAAAAABHP/AAAAAAAABHP+mZmZmZmZpHP+5mZmZmZmZK/////0cAAAAAAAAAAE6JRz/wAAAAAAAAiV2UXZSJSxBLAE5OiIiITolOSwCPlChLApB0lFKUfZRLAF2USwBhc4hOTl2UaACMElNlcXVlbmNlR3JvdXBTdGF0ZZSTlEsBSwCGlFKUfZROTk5LBk50lFKUaAUojAExlIh9lEsBaAkoaAwoaA9oEEsMQ0ACAAAAAAAAAIUAAAAAAAAAigEAAAAAAAAJAAAAAAAAAAUAAAAAAAAAOwEAAAAAAAAUAgAAAAAAABAAAAAAAAAAlHSUUpRoDChoD2gQSwxoFHSUUpRHAAAAAAAAAAAoSwJLhU2KAUsJSwVNOwFNFAJLEHSUSwBoG12UKEsCS4VNigFLCUsFTTsBTRQCSxBlXZR0lFKUc2giKEsBSwFHAAAAAAAAAABHAAAAAAAAAABHP/AAAAAAAABHP+mZmZmZmZpHP+5mZmZmZmZK/////0cAAAAAAAAAAE6JRz/wAAAAAAAAiV2UXZSJSxBLAE5OiIiITolOSwCPlChLApB0lFKUfZRLAV2USwFhc4hOTmgqaCxLAUsAhpRSlH2UTk5OSwhOdJRSlGgFKIwBMpSIfZRLAmgJKGgMKGgPaBBLDEMwAgAAAAAAAACFAAAAAAAAACwDAAAAAAAACQAAAAAAAAC+BQAAAAAAABAAAAAAAAAAlHSUUpRoDChoD2gQSwxoFHSUUpRHAAAAAAAAAAAoSwJLhU0sA0sJTb4FSxB0lEsAaBtdlChLAkuFTSwDSwlNvgVLEGVdlHSUUpRzaCIoSwFLAUcAAAAAAAAAAEcAAAAAAAAAAEc/8AAAAAAAAEc/6ZmZmZmZmkc/7mZmZmZmZkr/////RwAAAAAAAAAATolHP/AAAAAAAACJXZRdlIlLEEsATk6IiIhOiU5LAI+UKEsCkHSUUpR9lEsCXZRLAmFziE5OaCpoLEsBSwCGlFKUfZROTk5LBk50lFKUaAUojAEzlIh9lEsDaAkoaAwoaA9oEEsMQzACAAAAAAAAAIUAAAAAAAAA8wEAAAAAAAAJAAAAAAAAAE8SAAAAAAAAEAAAAAAAAACUdJRSlGgMKGgPaBBLDGgUdJRSlEcAAAAAAAAAAChLAkuFTfMBSwlNTxJLEHSUSwBoG12UKEsCS4VN8wFLCU1PEksQZV2UdJRSlHNoIihLAUsBRwAAAAAAAAAARwAAAAAAAAAARz/wAAAAAAAARz/pmZmZmZmaRz/uZmZmZmZmSv////9HAAAAAAAAAABOiUc/8AAAAAAAAIldlF2UiUsQSwBOToiIiE6JTksAj5QoSwKQdJRSlH2USwNdlEsDYXOITk5oKmgsSwFLAIaUUpR9lE5OTksGTnSUUpRlXZRdlF2USwBLAEsETksBXZROTnSUUpQu'
    # input_req = SpecDecodeDoc(no_spec=True, model_req=model_req)
    # llm_generate(input_req)
    opea_microservices["opea_service@spec_decode_draft_vllm"].start()
