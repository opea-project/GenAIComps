<div align="center">

# Generative AI Comps

<p align="center">
<b>Build Enterprise AI applications with microservice architecture</b>
</p>

<div align="left">

This project enables the creation of Enterprise AI services through microservices, streamlining the process of scaling and deployment to production. It abstracts away infrastructure complexities, facilitating the seamless development and deployment of Enterprise AI services.


## GenAIComps

GenAIComps provides a suite of microservices, leveraging a service composer to assemble a mega-service tailored for real-world Enterprise AI applications, all rigorously validated on Intel platforms.

Its advantages include:

- Streamlined containerization and orchestration of services and models, ensuring concurrency and scalability.
- Deployment of applications composed of multiple microservices, each capable of independent containerization and scaling.
- Effortless transition from local development to serving via the mega-service, followed by seamless readiness for production using Docker containers.

![Architecture](https://i.imgur.com/SuPqzOi.png)

## MicroService

Deploying AI applications using a microservice architecture has several advantages: service independently built and deployed distributed, easier main system integration, simpler testing, and reusable code components. We provides the below microservices, uses can follow the microservice design pattern to contribute your micorservice.

<table>
	<tbody>
		<tr>
			<td>MicroService</td>
            <td>Framework</td>
			<td>Model</td>
			<td>Serving</td>
			<td>HW</td>
			<td>Description</td>
		</tr>
		<tr>
			<td><a href="./comps/embeddings/README.md">Embedding</a></td>
            <td><a href="https://www.langchain.com">LangChain</a></td>
			<td><a href="https://huggingface.co/BAAI/bge-large-en-v1.5">BAAI/bge-large-en-v1.5</a></td>
			<td><a href="https://github.com/huggingface/tei-gaudi">TEI-Habana</a></td>
			<td>Gaudi2</td>
			<td>Embedding Microservice on Gaudi</td>
		</tr>
		<tr>
			<td><a href="./comps/embeddings/README.md">Embedding</a></td>
            <td><a href="https://www.langchain.com">LangChain</a></td>
			<td><a href="https://huggingface.co/BAAI/bge-base-en-v1.5">BAAI/bge-base-en-v1.5</a></td>
			<td><a href="https://github.com/huggingface/text-embeddings-inference">TEI</a></td>
			<td>Xeon</td>
			<td>Embedding Microservice on Xeon CPU</td>
		</tr>
		<tr>
			<td><a href="./comps/retrievers/README.md">Retriever</a></td>
			<td><a href="https://www.langchain.com">LangChain</a></td>
			<td><a href="https://huggingface.co/BAAI/bge-base-en-v1.5">BAAI/bge-base-en-v1.5</a></td>
			<td><a href="https://github.com/huggingface/text-embeddings-inference">TEI</a></td>
			<td>Xeon</td>
			<td>Retriever Microservice on Xeon CPU</td>
		</tr>
		<tr>
			<td><a href="./comps/reranks/README.md">Reranking</a></td>
            <td><a href="https://www.langchain.com">LangChain</a></td>
			<td><a href="https://huggingface.co/BAAI/bge-reranker-large">BAAI/bge-reranker-large</a></td>
			<td><a href="https://github.com/huggingface/tei-gaudi">TEI-Habana</a></td>
			<td>Gaudi2</td>
			<td>Reranking Microservice on Gaudi</td>
		</tr>
		<tr>
			<td><a href="./comps/reranks/README.md">Reranking</a></td>
            <td><a href="https://www.langchain.com">LangChain</a></td>
			<td><a href="https://huggingface.co/BAAI/bge-reranker-base">BBAAI/bge-reranker-base</a></td>
			<td><a href="https://github.com/huggingface/text-embeddings-inference">TEI</a></td>
			<td>Xeon</td>
			<td>Reranking Microservice on Xeon CPU</td>
		</tr>
		<tr>
			<td><a href="./comps/llms/README.md">LLM</a></td>
            <td><a href="https://www.langchain.com">LangChain</a></td>
			<td><a href="https://huggingface.co/Intel/neural-chat-7b-v3-3">Intel/neural-chat-7b-v3-3</a></td>
			<td><a href="https://github.com/huggingface/tgi-gaudi">TGI Gaudi</a></td>
			<td>Gaudi</td>
			<td>LLM Microservice on Gaudi</td>
		</tr>
		<tr>
			<td><a href="./comps/llms/README.md">LLM</a></td>
            <td><a href="https://www.langchain.com">LangChain</a></td>
			<td><a href="https://huggingface.co/Intel/neural-chat-7b-v3-3">Intel/neural-chat-7b-v3-3</a></td>
			<td><a href="https://github.com/huggingface/text-generation-inference">TGI</a></td>
			<td>Xeon</td>
			<td>LLM Microservice on Xeon CPU</td>
		</tr>
	</tbody>
</table>


## MegaService