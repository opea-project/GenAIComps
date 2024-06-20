# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import triton_python_backend_utils as pb_utils
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {"name": "INPUT0", "data_type": "TYPE_UINT8", "dims": [-1, -1]},
        ]
        outputs = [
            {"name": "OUTPUT0", "data_type": "TYPE_UINT8", "dims": [-1, -1, -1, 3]},
        ]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config["input"]:
            input_names.append(input["name"])
        for output in config["output"]:
            output_names.append(output["name"])

        return auto_complete_model_config

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        self.model_name = "stabilityai/stable-diffusion-2-base"
        self.scheduler = GaudiDDIMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        self.pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            self.model_name,
            scheduler=self.scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config="Habana/stable-diffusion-2",
        )

        output0_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        print("Initialized...")

    def execute(self, requests):
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            np_0 = in_0.as_numpy()
            to_string = lambda x: str(x.tobytes(), "utf8")
            prompts = list(np.apply_along_axis(to_string, 1, np_0))

            outputs = self.pipeline(
                prompt=prompts,
                num_images_per_prompt=1,
                height=768,
                width=768,
                batch_size=len(prompts),
            )
            arrs = []
            for image in outputs.images:
                arrs.append(np.array(image))

            out_arr = np.stack(arrs)
            ot = pb_utils.Tensor("OUTPUT0", out_arr.astype(self.output0_dtype))
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        ot,
                    ],
                )
            )

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        print("Cleaning up...")


if __name__ == "__main__":
    m = TritonPythonModel()
    m.initialize({})
    m.execute([1])
    m.finalize()
