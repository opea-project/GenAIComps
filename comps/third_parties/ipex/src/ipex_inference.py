# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Most of code originates from https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/llm/inference/single_instance/run_generation.py
# Refactored to FastAPI app for serving the model

import argparse
import logging
import os
import re
from typing import Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoConfig, TextStreamer

logger = logging.getLogger(__name__)

try:
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model

except ImportError:
    pass

from openai_protocol import ChatCompletionRequest
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    T5ForConditionalGeneration,
    WhisperForConditionalGeneration,
)

# supported models
MODEL_CLASSES = {
    "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
    "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
    "llama": (AutoModelForCausalLM, AutoTokenizer),
    "mllama": (MllamaForConditionalGeneration, AutoProcessor),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "falcon": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    "codegen": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan2": (AutoModelForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "chatglm": (AutoModelForCausalLM, AutoTokenizer),
    "gptbigcode": (AutoModelForCausalLM, AutoTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "mixtral": (AutoModelForCausalLM, AutoTokenizer),
    "mistral": (AutoModelForCausalLM, AutoTokenizer),
    "mpt": (AutoModelForCausalLM, AutoTokenizer),
    "stablelm": (AutoModelForCausalLM, AutoTokenizer),
    "qwen": (AutoModelForCausalLM, AutoTokenizer),
    "git": (AutoModelForCausalLM, AutoProcessor),
    "yuan": (AutoModelForCausalLM, AutoTokenizer),
    "phi-3": (AutoModelForCausalLM, AutoTokenizer),
    "phi-4-multimodal": (AutoModelForCausalLM, AutoProcessor),
    "phio": (AutoModelForCausalLM, AutoProcessor),
    "phi": (AutoModelForCausalLM, AutoTokenizer),
    "whisper": (WhisperForConditionalGeneration, AutoProcessor),
    "maira2": (AutoModelForCausalLM, AutoProcessor),
    "maira-2": (AutoModelForCausalLM, AutoProcessor),
    "jamba": (AutoModelForCausalLM, AutoTokenizer),
    "deepseek-v2": (AutoModelForCausalLM, AutoTokenizer),
    "deepseek-v3": (AutoModelForCausalLM, AutoTokenizer),
    "deepseekv2": (AutoModelForCausalLM, AutoTokenizer),
    "deepseekv3": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

try:
    from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

    MODEL_CLASSES["llava"] = (LlavaLlamaForCausalLM, AutoTokenizer)
except ImportError:
    pass


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

model = None
tokenizer = None
image_processor = None
model_type = ""
amp_enabled = False
amp_dtype = ""
config = None


def load_model(
    model_id: str,
    dtype: str = "bfloat16",
    ipex: bool = False,
    torch_compile: bool = False,
    deployment_mode: bool = False,
    cache_weight_for_large_batch: bool = False,
    backend: str = "ipex",
    vision_text_model: bool = False,
    config_file: str = None,
    kv_cache_dtype: str = "auto",
    prompt: str = None,
    input_tokens: int = 32,
    max_new_tokens: int = 32,
    **kwargs,
):
    global model, tokenizer, image_processor, config, model_type, amp_enabled, amp_dtype
    """Loads the model and processor.

    Args:
        model_id: Model ID or path.
        dtype: Data type (bfloat16/float32).
        ipex: Enable Intel Extension for PyTorch optimization.
        torch_compile: Enable torch.compile for performance optimization.
        deployment_mode: Optimize for deployment scenarios.
        cache_weight_for_large_batch: Enable weight caching for large batch inference.
        backend: Backend for torch.compile.

    Returns:
        ModelWrapper: A wrapper containing the model, tokenizer, and processor.
    """
    if ipex:
        import intel_extension_for_pytorch as ipex

        torch._C._jit_set_texpr_fuser_enabled(False)
        try:
            ipex._C.disable_jit_linear_repack()
        except Exception:
            pass

    # dtype
    amp_enabled = True if dtype != "float32" else False
    amp_dtype = getattr(torch, dtype)

    # load model
    model_type = next((x for x in MODEL_CLASSES.keys() if x in model_id.lower()), "auto")

    if model_type == "llama" and vision_text_model:
        model_type = "mllama"
    if model_type in ["maira-2", "deepseek-v2", "deepseek-v3"]:
        model_type = model_type.replace("-", "")
    model_class = MODEL_CLASSES[model_type]
    if config_file is None:
        if model_type == "chatglm":
            # chatglm modeling is from remote hub and its torch_dtype in config.json need to be overridden
            config = AutoConfig.from_pretrained(
                model_id,
                torchscript=deployment_mode,
                trust_remote_code=True,
                torch_dtype=amp_dtype,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_id,
                torchscript=deployment_mode,
                trust_remote_code=True,
            )
    else:
        config = AutoConfig.from_pretrained(
            config_file,
            torchscript=deployment_mode,
            trust_remote_code=True,
            torch_dtype=amp_dtype,
        )

    if kv_cache_dtype == "auto":
        kv_cache_dtype = None
    elif kv_cache_dtype == "fp8_e5m2":
        kv_cache_dtype = torch.float8_e5m2
    config.kv_cache_dtype = kv_cache_dtype

    if not hasattr(config, "text_max_length") and prompt is None:
        config.text_max_length = int(input_tokens) + int(max_new_tokens)
    if model_type == "mpt" and prompt is None:
        config.max_seq_len = int(input_tokens) + int(max_new_tokens)
    if model_type == "whisper":
        config.text_max_length = config.max_source_positions + config.max_target_positions
    if model_type == "jamba":
        config.use_mamba_kernels = False

    if not hasattr(config, "lm_head_generation"):
        config.lm_head_generation = True
    if model_type == "maira2" and not hasattr(config.text_config, "lm_head_generation"):
        config.text_config.lm_head_generation = True
    if model_type != "llava":
        model = model_class[0].from_pretrained(
            model_id,
            torch_dtype=amp_dtype,
            config=config,
            low_cpu_mem_usage=True if model_type != "maira2" else False,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        tokenizer = model_class[1].from_pretrained(model_id, trust_remote_code=True)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_id)
    model = model.eval()
    model = model.to(memory_format=torch.channels_last)

    # to ipex
    if ipex:
        model = ipex.llm.optimize(
            model.eval(),
            dtype=amp_dtype,
            inplace=True,
            deployment_mode=deployment_mode,
            cache_weight_for_large_batch=cache_weight_for_large_batch,
        )
    if torch_compile:
        if deployment_mode:
            raise SystemExit(
                "[ERROR] deployment_mode cannot co-work with torch.compile, please set deployment_mode"
                " to False if want to use torch.compile."
            )
        model.forward = torch.compile(model.forward, dynamic=True, backend=backend)


def predict(
    model_id: str,
    prompt: str = None,
    input_tokens: int = 32,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    greedy: bool = True,
    streaming: bool = True,
    image_url: str = None,
    audio_file: str = None,
    input_mode: int = 0,
    **kwargs,
):
    global model, tokenizer, image_processor, model_type, amp_enabled, amp_dtype, config

    if model_type in ["phio", "phi-4-multimodal"]:
        if model_type == "phi-4-multimodal":
            model_type = "phio"
        prompt = prompt
        _COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN = r"<\|image_\d+\|>"
        _COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN = r"<\|audio_\d+\|>"
        image_in_prompt = len(re.findall(_COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN, prompt))
        audio_in_prompt = len(re.findall(_COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN, prompt))
        is_vision = image_in_prompt > 0
        is_speech = audio_in_prompt > 0
        audio_batch_size = batch_size
        if is_vision:
            assert (
                image_in_prompt == batch_size
            ), "Prompt is invalid. For multiple images, the user needs to insert multiple image placeholders in the prompt as below: \
                <|user|><|image_1|><|image_2|><|image_3|>Summarize the content of the images.<|end|><|assistant|>"
        if is_speech:
            if not is_vision:
                assert (
                    audio_in_prompt == batch_size
                ), "Prompt is invalid. For multiple audios, the user needs to insert multiple audio placeholders in the prompt as below: \
                    <|user|><|audio_1|><|audio_2|><|audio_3|>Transcribe the audio clip into text.<|end|><|assistant|>"
            else:
                audio_batch_size = audio_in_prompt
        if not is_vision and not is_speech:
            config.input_mode = 0
        elif is_vision and not is_speech:
            config.input_mode = 1
        elif not is_vision and is_speech:
            config.input_mode = 2
        else:
            config.input_mode = 3

        assert config.input_mode == int(
            input_mode
        ), "Input mode in prompt is not consistent with the input mode in the command line."

    num_beams = 1 if greedy else 4
    # generate args
    if streaming:
        streamer = TextStreamer(tokenizer)
    else:
        streamer = None
    generate_kwargs = dict(
        do_sample=True,
        temperature=0.9,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        streamer=streamer,
    )
    if re.search("gptbigcode", model.config.architectures[0], re.IGNORECASE):
        model_type = "gptbigcode"
    if re.search("gptneox", model.config.architectures[0], re.IGNORECASE):
        model_type = "gpt-neox"
    elif re.search("t5", model.config.architectures[0], re.IGNORECASE):
        generate_kwargs["max_length"] = generate_kwargs["max_new_tokens"]
        generate_kwargs.pop("max_new_tokens")
    elif re.search("git", model.config.architectures[0], re.IGNORECASE) or re.search(
        "llava", model.config.architectures[0], re.IGNORECASE
    ):
        from io import BytesIO

        import requests
        from PIL import Image

        model.config.batch_size = int(batch_size) * num_beams

        def load_image(image_file):
            if image_file.startswith("http://") or image_file.startswith("https://"):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_file).convert("RGB")
            return image

    elif re.search("mllama", model.config.architectures[0], re.IGNORECASE) or re.search(
        "phio", model.config.architectures[0], re.IGNORECASE
    ):
        from PIL import Image

        def load_image(image_file):
            if image_file.startswith("http://") or image_file.startswith("https://"):
                import requests

                raw_image = Image.open(requests.get(image_url, stream=True).raw)
            else:
                raw_image = Image.open(image_file)
            return raw_image

    elif re.search("maira2", model.config.architectures[0], re.IGNORECASE):
        import requests
        from PIL import Image

        def download_and_open(url: str) -> Image.Image:
            response = requests.get(url, headers={"User-Agent": "MAIRA-2"}, stream=True)
            return Image.open(response.raw)

    if re.search("llava", model.config.architectures[0], re.IGNORECASE):
        model_name = get_model_name_from_path(model_id)
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ("user", "assistant")
        else:
            roles = conv.roles
    if re.search("yuan", model.config.architectures[0], re.IGNORECASE) or re.search(
        "jamba", model.config.architectures[0], re.IGNORECASE
    ):
        model.config.batch_size = int(batch_size) * num_beams
    if re.search("phio", model.config.architectures[0], re.IGNORECASE):
        model.config.batch_size = int(batch_size) * num_beams
        model.config.audio_batch_size = audio_batch_size * num_beams
    if re.search("whisper", model.config.architectures[0], re.IGNORECASE):
        import librosa

        sample = librosa.load(audio_file, sr=16000)
    if re.search("phio", model.config.architectures[0], re.IGNORECASE):
        if config.input_mode in [2, 3]:
            import soundfile

            sample = soundfile.read(audio_file)
        else:
            sample = None

    if model_type == "git":
        import requests
        from PIL import Image

        prompt = Image.open(requests.get(image_url, stream=True).raw)
        generate_kwargs.pop("min_new_tokens", None)
    elif model_type == "llava":
        if prompt is not None:
            prompt = prompt
        image = load_image(image_url)
        image = [image] * batch_size
        if model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif model_type == "whisper":
        prompt = sample[0]
        generate_kwargs.pop("min_new_tokens", None)
    elif model_type == "maira2":
        prompt = prompt
        sample = download_and_open(image_url)
        process_input_func = (
            tokenizer.process_reporting_input
            if hasattr(tokenizer, "process_reporting_input")
            else tokenizer.format_and_preprocess_reporting_input
        )
    elif model_type == "phio":
        prompt = prompt
    else:
        if model_type == "mllama":
            raw_image = load_image(image_url)
            raw_image = [raw_image] * batch_size
            inputs = tokenizer(raw_image, prompt, return_tensors="pt")
            input_size = inputs["input_ids"].size(dim=1)
        else:
            input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
        print("---- Prompt size:", input_size)

    # start
    num_iter = 1
    prompt = [prompt] * batch_size
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
        for _ in range(num_iter):
            if model_type == "llava":
                input_ids = torch.stack(
                    [tokenizer_image_token(pmt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for pmt in prompt]
                )
                image_tensor = [
                    image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(amp_dtype) for img in image
                ]
                output = model.generate(input_ids, images=image_tensor, **generate_kwargs)
            elif model_type == "git":
                input_ids = tokenizer(images=prompt, return_tensors="pt").pixel_values
                output = model.generate(pixel_values=input_ids, **generate_kwargs)
            elif model_type == "whisper":
                input_ids = tokenizer(prompt, sampling_rate=16000, return_tensors="pt").input_features
                output = model.generate(input_ids, **generate_kwargs)
            elif model_type == "mllama":
                raw_image = load_image(image_url)
                raw_image = [raw_image] * batch_size
                inputs = tokenizer(raw_image, prompt, return_tensors="pt")
                input_ids = inputs["input_ids"]
                output = model.generate(**inputs, **generate_kwargs)
            elif model_type == "maira2":
                processed_inputs = process_input_func(
                    current_frontal=sample,
                    current_lateral=None,
                    prior_frontal=None,
                    indication=None,
                    technique=None,
                    comparison=None,
                    prior_report=None,
                    return_tensors="pt",
                    get_grounding=False,
                )
                input_ids = processed_inputs["input_ids"]
                output = model.generate(**processed_inputs, **generate_kwargs)
            elif model_type == "phio":
                raw_image = load_image(image_url) if is_vision else None
                raw_image = [raw_image] * batch_size
                samples = [sample] * audio_batch_size
                inputs = tokenizer(
                    text=prompt[0],
                    images=raw_image if is_vision else None,
                    audios=samples if is_speech else None,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"]
                output = model.generate(**inputs, **generate_kwargs)
            else:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = model.generate(input_ids, **generate_kwargs)
            gen_ids = output
            gen_text = tokenizer.batch_decode(
                (gen_ids[:, input_ids.shape[1] :] if model_type in ["llava", "maira2", "phio"] else gen_ids),
                skip_special_tokens=True,
            )
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o if model.config.model_type in ["t5", "whisper"] else o - i
                for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            print(gen_text, total_new_tokens, flush=True)
            return gen_text


@app.post("/v1/chat/completions")
async def llm_generate(input: Union[ChatCompletionRequest]) -> StreamingResponse:
    try:
        model_id = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
        message = None
        if isinstance(input.messages, str):
            message = input.messages
        else:  # List[Dict]
            for input_data in input.messages:
                if "role" in input_data and input_data["role"] == "user" and "content" in input_data:
                    message = input_data["content"]
        if input.stream:

            async def stream_generator():
                for output in predict(
                    model_id=model_id,
                    prompt=message,
                    input_tokens=2048,
                    max_new_tokens=input.max_tokens,
                    batch_size=1,
                    input_mode=0,
                    greedy=True,
                    streaming=True,
                    image_url=None,
                    audio_file=None,
                ):
                    yield f"data: {output}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            response = predict(
                model_id=model_id,
                prompt=message,
                input_tokens=2048,
                max_new_tokens=input.max_tokens,
                batch_size=1,
                input_mode=0,
                greedy=True,
                streaming=False,
                image_url=None,
                audio_file=None,
            )
            return response

    except Exception as e:
        logger.error(f"Error during IPEX LLM inference: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8688)

    args = parser.parse_args()
    model_id = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
    load_model(
        model_id=model_id,
        dtype="bfloat16",
        ipex=False,
        torch_compile=False,
        deployment_mode=False,
        cache_weight_for_large_batch=False,
        backend="ipex",
        vision_text_model=False,
        config_file=None,
        kv_cache_dtype="auto",
        input_tokens=2048,
        max_new_tokens=4096,
    )

    uvicorn.run(app, host=args.host, port=args.port)
