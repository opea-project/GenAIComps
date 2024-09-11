# Trust and Safety with LLM

The Guardrails service enhances the security of LLM-based applications by offering a suite of microservices designed to ensure trustworthiness, safety, and security.

| MicroService                                         | Description                                                                                                              |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Llama Guard](./llama_guard/langchain/README.md)     | Provides guardrails for inputs and outputs to ensure safe interactions                                                   |
| [PII Detection](./pii_detection/README.md)           | Detects Personally Identifiable Information (PII) and Business Sensitive Information (BSI)                               |
| [Toxicity Detection](./toxicity_detection/README.md) | Detects Toxic language (rude, disrespectful, or unreasonable language that is likely to make someone leave a discussion) |
| [Factuality](./factuality/README.md) | evaluate the outputs of LLMs against the context of the prompts.Generates a score indicating how factual the response is |
| [Prompt Injection](./prompt_injection/README.md) | Detects and prevents prompt injection attacks that attempt to manipulate the AI’s behavior or responses by injecting malicious or misleading input. This helps maintain the integrity and safety of AI interactions. |

Additional safety-related microservices will be available soon.
