FROM ubuntu:latest
RUN apt update && apt -y install wget git cmake build-essential
RUN wget https://go.dev/dl/go1.23.2.linux-amd64.tar.gz && rm -rf /usr/local/go && tar -C /usr/local -xzf go1.23.2.linux-amd64.tar.gz
ENV PATH="$PATH:/usr/local/go/bin"
WORKDIR /ollama
RUN git clone https://github.com/ollama/ollama.git .
RUN cd /ollama && OLLAMA_CUSTOM_CPU_DEFS="-DGGML_AVX=on -DGGML_AVX2=on -DGGML_F16C=on -DGGML_FMA=on -DGGML_AVX512=on -DGGML_AVX512_VNNI=on -DGGML_AVX512_VBMI=on " go generate ./... && go build .
ENV PATH="$PATH:/ollama"
COPY entrypoint.sh .
ENTRYPOINT ["bash", "entrypoint.sh"]