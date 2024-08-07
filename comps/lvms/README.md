# Language Vision Model (LVM) Microservice

This microservice, designed for Large Vision Model (LVM) Inference, processes input consisting of a query string and/or image prompts. It constructs a prompt based on the input, which is then used to perform inference with a large vision model (e.g., LLaVA). The service delivers the inference results as output.

A prerequisite for using this microservice is that users must have an LVM service (transformers or Prediction Guard) already running. Overall, this microservice offers a streamlined way to integrate large vision model inference into applications, requiring minimal setup from the user. This allows for the seamless processing of text and image prompts to generate intelligent, context-aware responses.

## Getting started with transformers + fastAPI services

The [transformers](transformers) directory contains instructions for running services that serve predictions from a LLaVA LVM. Two services must be spun up to run the LVM:

1. A LLaVA model server under [transformers/llava](transformers/llava)
2. An OPEA LVM service under [transformers](transformers)

See [transformers/README.md](transformers/README.md) for more information.

## Getting started with Prediction Guard

The [predictionguard](predictionguard) directory contains instructions for running a single service that serves predictions from a LLaVA LVM via the Prediction Guard framework hosted on Intel Tiber Developer Cloud (ITDC). See [predictionguard](predictionguard) for more information.
