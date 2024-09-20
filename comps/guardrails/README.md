# Trust and Safety with LLM

The Guardrails service enhances the security of LLM-based applications by offering a suite of microservices designed to ensure trustworthiness, safety, and security.

| MicroService                                         | Description                                                                                                              |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Llama Guard](./llama_guard/langchain/README.md)     | Provides guardrails for inputs and outputs to ensure safe interactions                                                   |
| [PII Detection](./pii_detection/README.md)           | Detects Personally Identifiable Information (PII) and Business Sensitive Information (BSI)                               |
| [Toxicity Detection](./toxicity_detection/README.md) | Detects Toxic language (rude, disrespectful, or unreasonable language that is likely to make someone leave a discussion) |
| [Bias Detection](./bias_detection/README.md)         | Detects Biased language (framing bias, epistemological bias, and demographic bias)                                       |
| [Prediction Guard PII Detection](./pii_detection/predictionguard/README.md) | Detects and verifies or replaces Personally Identifiable Information (PII).                      |
| [Prediction Guard Toxicity Detection](./toxicity_detection/predictionguard/README.md) | Detects Toxic language (rude, disrespectful, or unreasonable language)                  |
| [Prediction Guard Factuality Check](./factuality/predictionguard/README.md) | Evaluates the factual consistency of the model responses with the input context                   |
| [Prediction Guard Injection Check](./prompt_injection/predictionguard/README.md) | Detects and mitigates prompt injection attacks to ensure safe model interactions             |



Additional safety-related microservices will be available soon.
