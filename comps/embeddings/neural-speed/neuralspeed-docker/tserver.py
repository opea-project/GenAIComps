from typing import Any, List
from mosec import Server, Worker, get_logger
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from neural_speed import Model

logger = get_logger()

INFERENCE_BATCH_SIZE = 32
INFERENCE_WORKER_NUM = 1


class Request(Struct, kw_only=True):
    query: str
    docs: List[str]


class Response(Struct, kw_only=True):
    scores: List[float]


class Inference(TypedMsgPackMixin, Worker):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("/home/zhenzhong/model/bce-reranker-base_v1")
        self.model = Model()
        self.model.init_from_bin(
            "bert",
            "/home/zhenzhong/repo/opea/neural-speed/scripts/mosec/rerank-q8j-g-1-cint8.bin",
            batch_size=INFERENCE_BATCH_SIZE,
            n_ctx=514,
        )

    def forward(self, data: Request) -> Request:
        inp = [[data.query, doc] for doc in data.docs]
        inputs = self.tokenizer(inp, padding=True, truncation=True, max_length=512, return_tensors="pt")
        ns_outputs = self.model(
            inputs.input_ids,
            reinit=True,
            logits_all=True,
            continuous_batching=False,
            ignore_padding=True,
        )
        return Response(scores=ns_outputs.tolist())


if __name__ == "__main__":
    server = Server()
    server.append_worker(Inference, num=INFERENCE_WORKER_NUM)
    server.run()
