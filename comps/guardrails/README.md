# Trust and Safety with LLM

The Guardrails service enhances the security of LLM-based applications by offering a suite of microservices designed to ensure trustworthiness, safety, and security.

| MicroService                                                   | Description                                                                                                              | Contributors |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------ |
| [Guardrails](./src/guardrails/README.md)           | Provides general guardrails for inputs and outputs to ensure safe interactions using [Llama Guard](./src/guardrails/README.md#LlamaGuard) or [WildGuard](./src/guardrails/README.md#WildGuard)                     | Intel  |
| [PII Detection](./src/pii_detection/README.md)                 | Detects Personally Identifiable Information (PII) and Business Sensitive Information (BSI)                               | Prediction Guard |
| [Toxicity Detection](./src/toxicity_detection/README.md)       | Detects Toxic language (rude, disrespectful, or unreasonable language that is likely to make someone leave a discussion) | Intel, Prediction Guard            |
| [Bias Detection](./src/bias_detection/README.md)               | Detects Biased language (framing bias, epistemological bias, and demographic bias)                                       | Intel            |
| [Prompt Injection Detection](./src/prompt_injection/README.md) | Detects malicious prompts causing the system running an LLM to execute the attacker’s intentions                         | Intel, PredictionGuard |
| [Factuality Alignment](.src/factuality_alignment/README.md)    | Detects hallucination by checking for factual consistency between two text passages                         | PredictionGuard |
| [Hallucination](.src/hallucination_detection/README.md)        | Detects hallucination given a text document, question and answer                         | Intel |


Additional safety-related microservices will be available soon.
