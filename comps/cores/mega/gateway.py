
from .constants import MegaServiceEndpoint, ServiceRoleType, ServiceType
from .micro_service import MicroService

from ..proto.api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)

class Gateway:
    def __init__(self, megaservice, host, port, endpoint, input_datatype, output_datatype):
        self.megaservice = megaservice
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.input_datatype = input_datatype
        self.output_datatype = output_datatype
        self.service = MicroService(
            service_role=ServiceRoleType.MEGASERVICE,
            service_type=ServiceType.GATEWAY,
            host=self.host,
            port=self.port,
            endpoint=self.endpoint,
            input_datatype=self.input_datatype,
            output_datatype=self.output_datatype,
        )
        self.define_routes()
        self.service.start()

    def define_routes(self):
        self.gateway.app.router.add_api_route(self.endpoint, self.handle_request, methods=["POST"])
        self.gateway.app.router.add_api_route(str(MegaServiceEndpoint.LIST_SERVICE), self.list_service, methods=["GET"])
        self.gateway.app.router.add_api_route(
            str(MegaServiceEndpoint.LIST_PARAMETERS), self.list_parameter, methods=["GET"]
        )

    def handle_request(self, request):
        raise NotImplementedError("Subclasses must implement this method")

    def list_service(self):
        raise NotImplementedError("Subclasses must implement this method")

    def list_parameter(self):
        raise NotImplementedError("Subclasses must implement this method")

class ChatQnAGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(megaservice, host, port, str(MegaServiceEndpoint.CHAT_QNA), ChatCompletionRequest, ChatCompletionResponse)

    async def handle_request(self, request):
        data = await request.json()
        chat_request = ChatCompletionRequest.parse_obj(data)
        if isinstance(chat_request.messages, str):
            prompt = chat_request.messages
        else:
            for message in chat_request.messages:
                text_list = [item["text"] for item in message["content"] if item["type"] == "text"]
                prompt = "\n".join(text_list)
        self.megaservice.schedule(initial_inputs={"text": prompt})
        last_node = self.megaservice.all_leaves()[-1]
        response = self.megaservice.result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="chatqna", choices=choices, usage=usage)

    def list_service(self):
        response = {}
        for node in self.all_leaves():
            response = {self.services[node].description: self.services[node].endpoint_path}
        return response

    def list_parameter(self):
        pass
