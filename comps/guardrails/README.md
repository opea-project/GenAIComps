# Trust and Safety with LLM

The Guardrails service enhances the security of LLM-based applications by offering a suite of microservices designed to ensure trustworthiness, safety, and security.

| MicroService                                                   | Description                                                                                                                                                                                    | Contributors            |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| [Bias Detection](./src/bias_detection/README.md)               | Detects Biased language (framing bias, epistemological bias, and demographic bias)                                                                                                             | Intel                   |
| [Factuality Alignment](./src/factuality_alignment/README.md)   | Detects hallucination by checking for factual consistency between two text passages                                                                                                            | PredictionGuard         |
| [Guardrails](./src/guardrails/README.md)                       | Provides general guardrails for inputs and outputs to ensure safe interactions using [Llama Guard](./src/guardrails/README.md#LlamaGuard) or [WildGuard](./src/guardrails/README.md#WildGuard) | Intel                   |
| [Hallucination](./src/hallucination_detection/README.md)       | Detects hallucination given a text document, question and answer                                                                                                                               | Intel                   |
| [PII Detection](./src/pii_detection/README.md)                 | Detects Personally Identifiable Information (PII) and Business Sensitive Information (BSI)                                                                                                     | Prediction Guard        |
| [Prompt Injection Detection](./src/prompt_injection/README.md) | Detects malicious prompts causing the system running an LLM to execute the attacker’s intentions                                                                                               | Intel, PredictionGuard  |
| [Toxicity Detection](./src/toxicity_detection/README.md)       | Detects Toxic language (rude, disrespectful, or unreasonable language that is likely to make someone leave a discussion)                                                                       | Intel, Prediction Guard |

Additional safety-related microservices will be available soon.
