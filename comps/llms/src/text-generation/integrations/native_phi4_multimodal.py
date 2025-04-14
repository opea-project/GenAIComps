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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

sys.path.append("/test/GenAIComps/")

import os
import threading
import time

import habana_frameworks.torch.core as htcore
import soundfile
import torch
from langchain_core.prompts import PromptTemplate
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from comps import CustomLogger, GeneratedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import ChatCompletionRequest

from .template import ChatTemplate

logger = CustomLogger("opea_textgen_native_phi_multimodal")
logflag = os.getenv("LOGFLAG", False)

MODEL_NAME = os.getenv("LLM_MODEL_ID", "microsoft/Phi-4-multimodal-instruct")

model = None
processor = None
generation_config = None
initialization_lock = threading.Lock()
initialized = False

kwargs = {}
kwargs["torch_dtype"] = torch.bfloat16

user_prompt = "<|user|>"
assistant_prompt = "<|assistant|>"
prompt_suffix = "<|end|>"
IMAGE_SPECIAL = "<|endoftext10|>"
AUDIO_SPECIAL = "<|endoftext11|>"
sample_prompt = f"{user_prompt}what is the answer for 1+1? Explain it.{prompt_suffix}{assistant_prompt}"
if logflag:
    logger.info(f">>> Prompt\n{sample_prompt}")

generation_config = GenerationConfig.from_pretrained(MODEL_NAME, "generation_config.json")

# generation_config.max_new_tokens = args.max_new_tokens
# generation_config.use_cache = args.use_kv_cache
generation_config.static_shapes = False  # There's a list of models optimized with static shapes
generation_config.bucket_size = -1
generation_config.bucket_internal = False
# generation_config.do_sample = args.do_sample
# generation_config.num_beams = args.num_beams
# generation_config.top_k = args.top_k
# generation_config.penalty_alpha = args.penalty_alpha
# generation_config.bad_words_ids = bad_words_ids
# generation_config.force_words_ids = force_words_ids
# generation_config.num_return_sequences = args.num_return_sequences
generation_config.trim_logits = True
generation_config.attn_softmax_bf16 = False
generation_config.limit_hpu_graphs = False
generation_config.clear_hpu_graphs_cache = False
generation_config.reuse_cache = False
generation_config.reduce_recompile = False
# if generation_config.reduce_recompile:
#     assert generation_config.bucket_size > 0
generation_config.use_flash_attention = False
generation_config.flash_attention_recompute = False
generation_config.flash_attention_causal_mask = False
generation_config.flash_attention_fast_softmax = False
# generation_config.trust_remote_code = args.trust_remote_code
generation_config.valid_sequence_lengths = None  # OkS
generation_config.attn_batch_split = False
generation_config.ignore_eos = None


def generate(
    query,
    image_path=None,
    audio_path=None,
    max_tokens=128,
):
    """Generates sequences from the input sentences and returns them."""
    query_prompt = f"{user_prompt}"
    images = None
    audios = None
    if image_path is not None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Provided {image_path} not found.")
        images = [Image.open(image_path)]
        query_prompt += f"{IMAGE_SPECIAL}"
    if audio_path is not None:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Provided {audio_path} not found.")
        audios = [soundfile.read(audio_path)]
        query_prompt += f"{AUDIO_SPECIAL}"
    query_prompt += f"{query}{prompt_suffix}{assistant_prompt}"
    logger.info(f"[llm - generate] starting to inference with prompt {query_prompt}")

    inputs = processor(query_prompt, images=images, audios=audios, return_tensors="pt").to("hpu:0")

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if logflag:
        logger.info(response)
    print(f">>> Response\n{response}")

    return response


def initialize():
    global model, processor, generation_config, initialized
    with initialization_lock:
        if not initialized:
            processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="sdpa",
            )
            model = model.to("hpu")
            if logflag:
                logger.info(processor.tokenizer)
                logger.info(f"model.config._attn_implementation: {model.config._attn_implementation}")
            logger.info("[llm] model and processor initialized.")

            # Must put after the models are downloaded because this has custom remote code that needs to be loaded first for the OH to load the override functions
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

            adapt_transformers_to_gaudi()

            logger.info("[llm - native] Ready to inference")
            res = generate(sample_prompt)
            logger.info(f"[llm - native] test result: {res}")
            initialized = True


@OpeaComponentRegistry.register("OpeaTextGenNativePhi4Multimodal")
class OpeaTextGenNativePhi4Multimodal(OpeaComponent):
    """A specialized OPEA TextGen component derived from OpeaComponent for interacting with LLM services based on native optimum habana."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.LLM.name.lower(), description, config)
        initialize()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaTextGenNativePhi4Multimodal health check failed.")
        else:
            logger.info("OpeaTextGenNativePhi4Multimodal health check success.")

    def check_health(self) -> bool:
        """Checks the health of the LLM service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """

        try:
            return initialized
        except Exception as e:
            logger.error(e)
            logger.error("Health check failed")
            return False

    async def invoke(self, input: ChatCompletionRequest):
        """Invokes the LLM service to generate output for the provided input.

        Args:
            input (ChatCompletionRequest): The input text(s).
        """

        message = None
        if isinstance(input.messages, str):
            message = input.messages
        else:  # List[Dict]
            for input_data in input.messages:
                if "role" in input_data and input_data["role"] == "user" and "content" in input_data:
                    message = input_data["content"]
                    if logflag:
                        logger.info(f"Get input text:\n {message}")
        if message is None:
            logger.error("Don't receive any input text, exit!")
            return GeneratedDoc(text=None, prompt=None)

        prompt = message
        prompt_template = None
        if input.chat_template:
            prompt_template = PromptTemplate.from_template(input.chat_template)
            input_variables = prompt_template.input_variables
        if prompt_template:
            if sorted(input_variables) == ["context", "question"]:
                prompt = prompt_template.format(question=message, context="\n".join(input.documents))
            elif input_variables == ["question"]:
                prompt = prompt_template.format(question=message)
            else:
                logger.info(f"{prompt_template} not used, we only support 2 input variables ['question', 'context']")
        else:
            if input.documents:
                prompt = ChatTemplate.generate_rag_prompt(message, input.documents)
        res = generate(prompt, image_path=input.image_path, audio_path=input.audio_path, max_tokens=input.max_tokens)

        if logflag:
            logger.info(f"[llm - native] inference result: {res}")
        return GeneratedDoc(text=res, prompt=message)
