# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from docarray import BaseDoc
from docarray.documents import AudioDoc
from docarray.typing import AudioUrl, ImageUrl
from pydantic import Field, NonNegativeFloat, PositiveInt, conint, conlist, field_validator


class TopologyInfo:
    # will not keep forwarding to the downstream nodes in the black list
    # should be a pattern string
    downstream_black_list: Optional[list] = []


class TextDoc(BaseDoc, TopologyInfo):
    text: Union[str, List[str]] = None


class Audio2text(BaseDoc, TopologyInfo):
    query: str = None


class FactualityDoc(BaseDoc):
    reference: str
    text: str


class ScoreDoc(BaseDoc):
    score: float


class PIIRequestDoc(BaseDoc):
    prompt: str
    replace: Optional[bool] = False
    replace_method: Optional[str] = "random"


class PIIResponseDoc(BaseDoc):
    detected_pii: Optional[List[dict]] = None
    new_prompt: Optional[str] = None


class MetadataTextDoc(TextDoc):
    metadata: Optional[Dict[str, Any]] = Field(
        description="This encloses all metadata associated with the textdoc.",
        default=None,
    )


class ImageDoc(BaseDoc):
    url: Optional[ImageUrl] = Field(
        description="The path to the image. It can be remote (Web) URL, or a local file path",
        default=None,
    )
    base64_image: Optional[str] = Field(
        description="The base64-based encoding of the image",
        default=None,
    )


class TextImageDoc(BaseDoc):
    image: ImageDoc = None
    text: TextDoc = None


MultimodalDoc = Union[
    TextDoc,
    ImageDoc,
    TextImageDoc,
]


class Base64ByteStrDoc(BaseDoc):
    byte_str: str


class DocSumDoc(BaseDoc):
    text: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None


class DocPath(BaseDoc):
    path: str
    chunk_size: int = 1500
    chunk_overlap: int = 100
    process_table: bool = False
    table_strategy: str = "fast"


class EmbedDoc(BaseDoc):
    text: Union[str, List[str]]
    embedding: Union[conlist(float, min_length=0), List[conlist(float, min_length=0)]]
    search_type: str = "similarity"
    k: PositiveInt = 4
    distance_threshold: Optional[float] = None
    fetch_k: PositiveInt = 20
    lambda_mult: NonNegativeFloat = 0.5
    score_threshold: NonNegativeFloat = 0.2
    constraints: Optional[Union[Dict[str, Any], List[Dict[str, Any]], None]] = None
    index_name: Optional[str] = None


class EmbedMultimodalDoc(EmbedDoc):
    # extend EmbedDoc with these attributes
    url: Optional[ImageUrl] = Field(
        description="The path to the image. It can be remote (Web) URL, or a local file path.",
        default=None,
    )
    base64_image: Optional[str] = Field(
        description="The base64-based encoding of the image.",
        default=None,
    )


class Audio2TextDoc(AudioDoc):
    url: Optional[AudioUrl] = Field(
        description="The path to the audio.",
        default=None,
    )
    model_name_or_path: Optional[str] = Field(
        description="The Whisper model name or path.",
        default="openai/whisper-small",
    )
    language: Optional[str] = Field(
        description="The language that Whisper prefer to detect.",
        default="auto",
    )


class SearchedDoc(BaseDoc):
    retrieved_docs: List[TextDoc]
    initial_query: str
    top_n: PositiveInt = 1

    class Config:
        json_encoders = {np.ndarray: lambda x: x.tolist()}


class SearchedMultimodalDoc(SearchedDoc):
    metadata: List[Dict[str, Any]]


class LVMSearchedMultimodalDoc(SearchedMultimodalDoc):
    max_new_tokens: conint(ge=0, le=1024) = 512
    top_k: int = 10
    top_p: float = 0.95
    typical_p: float = 0.95
    temperature: float = 0.01
    stream: bool = False
    repetition_penalty: float = 1.03
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead. We recommend that the template contains {context} and {question} for multimodal-rag on videos."
        ),
    )


class RerankedDoc(BaseDoc):
    reranked_docs: List[TextDoc]
    initial_query: str


class AnonymizeModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    hidden_names: Optional[List[str]] = None
    allowed_names: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    preamble: Optional[str] = None
    regex_patterns: Optional[List[str]] = None
    use_faker: Optional[bool] = None
    recognizer_conf: Optional[str] = None
    threshold: Optional[float] = None
    language: Optional[str] = None


class BanCodeModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None


class BanCompetitorsModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    competitors: List[str] = ["Competitor1", "Competitor2", "Competitor3"]
    model: Optional[str] = None
    threshold: Optional[float] = None
    redact: Optional[bool] = None


class BanSubstringsModel(BaseDoc):
    enabled: bool = False
    substrings: List[str] = ["backdoor", "malware", "virus"]
    match_type: Optional[str] = None
    case_sensitive: bool = False
    redact: Optional[bool] = None
    contains_all: Optional[bool] = None


class BanTopicsModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    topics: List[str] = ["violence", "attack", "war"]
    threshold: Optional[float] = None
    model: Optional[str] = None


class CodeModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    languages: List[str] = ["Java", "Python"]
    model: Optional[str] = None
    is_blocked: Optional[bool] = None
    threshold: Optional[float] = None


class GibberishModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None
    match_type: Optional[str] = None


class InvisibleText(BaseDoc):
    enabled: bool = False


class LanguageModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    valid_languages: List[str] = ["en", "es"]
    model: Optional[str] = None
    threshold: Optional[float] = None
    match_type: Optional[str] = None


class PromptInjectionModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None
    match_type: Optional[str] = None


class RegexModel(BaseDoc):
    enabled: bool = False
    patterns: List[str] = ["Bearer [A-Za-z0-9-._~+/]+"]
    is_blocked: Optional[bool] = None
    match_type: Optional[str] = None
    redact: Optional[bool] = None


class SecretsModel(BaseDoc):
    enabled: bool = False
    redact_mode: Optional[str] = None


class SentimentModel(BaseDoc):
    enabled: bool = False
    threshold: Optional[float] = None
    lexicon: Optional[str] = None


class TokenLimitModel(BaseDoc):
    enabled: bool = False
    limit: Optional[int] = None
    encoding_name: Optional[str] = None
    model_name: Optional[str] = None


class ToxicityModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None
    match_type: Optional[str] = None


class BiasModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None
    match_type: Optional[str] = None


class DeanonymizeModel(BaseDoc):
    enabled: bool = False
    matching_strategy: Optional[str] = None


class JSONModel(BaseDoc):
    enabled: bool = False
    required_elements: Optional[int] = None
    repair: Optional[bool] = None


class LanguageSameModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None


class MaliciousURLsModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None


class NoRefusalModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None
    match_type: Optional[str] = None


class NoRefusalLightModel(BaseDoc):
    enabled: bool = False


class ReadingTimeModel(BaseDoc):
    enabled: bool = False
    max_time: float = 0.5
    truncate: Optional[bool] = None


class FactualConsistencyModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    minimum_score: Optional[float] = None


class RelevanceModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    model: Optional[str] = None
    threshold: Optional[float] = None


class SensitiveModel(BaseDoc):
    enabled: bool = False
    use_onnx: bool = False
    entity_types: Optional[List[str]] = None
    regex_patterns: Optional[List[str]] = None
    redact: Optional[bool] = None
    recognizer_conf: Optional[str] = None
    threshold: Optional[float] = None


class URLReachabilityModel(BaseDoc):
    enabled: bool = False
    success_status_codes: Optional[List[int]] = None
    timeout: Optional[int] = None


class LLMGuardInputGuardrailParams(BaseDoc):
    anonymize: AnonymizeModel = AnonymizeModel()
    ban_code: BanCodeModel = BanCodeModel()
    ban_competitors: BanCompetitorsModel = BanCompetitorsModel()
    ban_substrings: BanSubstringsModel = BanSubstringsModel()
    ban_topics: BanTopicsModel = BanTopicsModel()
    code: CodeModel = CodeModel()
    gibberish: GibberishModel = GibberishModel()
    invisible_text: InvisibleText = InvisibleText()
    language: LanguageModel = LanguageModel()
    prompt_injection: PromptInjectionModel = PromptInjectionModel()
    regex: RegexModel = RegexModel()
    secrets: SecretsModel = SecretsModel()
    sentiment: SentimentModel = SentimentModel()
    token_limit: TokenLimitModel = TokenLimitModel()
    toxicity: ToxicityModel = ToxicityModel()


class LLMGuardOutputGuardrailParams(BaseDoc):
    ban_code: BanCodeModel = BanCodeModel()
    ban_competitors: BanCompetitorsModel = BanCompetitorsModel()
    ban_substrings: BanSubstringsModel = BanSubstringsModel()
    ban_topics: BanTopicsModel = BanTopicsModel()
    bias: BiasModel = BiasModel()
    code: CodeModel = CodeModel()
    deanonymize: DeanonymizeModel = DeanonymizeModel()
    json_scanner: JSONModel = JSONModel()
    language: LanguageModel = LanguageModel()
    language_same: LanguageSameModel = LanguageSameModel()
    malicious_urls: MaliciousURLsModel = MaliciousURLsModel()
    no_refusal: NoRefusalModel = NoRefusalModel()
    no_refusal_light: NoRefusalLightModel = NoRefusalLightModel()
    reading_time: ReadingTimeModel = ReadingTimeModel()
    factual_consistency: FactualConsistencyModel = FactualConsistencyModel()
    gibberish: GibberishModel = GibberishModel()
    regex: RegexModel = RegexModel()
    relevance: RelevanceModel = RelevanceModel()
    sensitive: SensitiveModel = SensitiveModel()
    sentiment: SentimentModel = SentimentModel()
    toxicity: ToxicityModel = ToxicityModel()
    url_reachability: URLReachabilityModel = URLReachabilityModel()
    anonymize_vault: Optional[List[Tuple]] = (
        None  # the only parameter not available in fingerprint. Used to transmit vault
    )


class LLMParamsDoc(BaseDoc):
    model: Optional[str] = None  # for openai and ollama
    query: str
    max_tokens: int = 1024
    max_new_tokens: PositiveInt = 1024
    top_k: PositiveInt = 10
    top_p: NonNegativeFloat = 0.95
    typical_p: NonNegativeFloat = 0.95
    temperature: NonNegativeFloat = 0.01
    frequency_penalty: NonNegativeFloat = 0.0
    presence_penalty: NonNegativeFloat = 0.0
    repetition_penalty: NonNegativeFloat = 1.03
    stream: bool = True
    language: str = "auto"  # can be "en", "zh"
    input_guardrail_params: Optional[LLMGuardInputGuardrailParams] = None
    output_guardrail_params: Optional[LLMGuardOutputGuardrailParams] = None

    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead. We recommend that the template contains {context} and {question} for rag,"
            "or only contains {question} for chat completion without rag."
        ),
    )
    documents: Optional[Union[List[Dict[str, str]], List[str]]] = Field(
        default=[],
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            '"title" and "text" keys.'
        ),
    )

    @field_validator("chat_template")
    def chat_template_must_contain_variables(cls, v):
        return v


class GeneratedDoc(BaseDoc):
    text: str
    prompt: str
    output_guardrail_params: Optional[LLMGuardOutputGuardrailParams] = None


class LLMParams(BaseDoc):
    model: Optional[str] = None
    max_tokens: int = 1024
    max_new_tokens: PositiveInt = 1024
    top_k: PositiveInt = 10
    top_p: NonNegativeFloat = 0.95
    typical_p: NonNegativeFloat = 0.95
    temperature: NonNegativeFloat = 0.01
    frequency_penalty: NonNegativeFloat = 0.0
    presence_penalty: NonNegativeFloat = 0.0
    repetition_penalty: NonNegativeFloat = 1.03
    stream: bool = True
    language: str = "auto"  # can be "en", "zh"
    index_name: Optional[str] = None

    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead. We recommend that the template contains {context} and {question} for rag,"
            "or only contains {question} for chat completion without rag."
        ),
    )


class RetrieverParms(BaseDoc):
    search_type: str = "similarity"
    k: PositiveInt = 4
    distance_threshold: Optional[float] = None
    fetch_k: PositiveInt = 20
    lambda_mult: NonNegativeFloat = 0.5
    score_threshold: NonNegativeFloat = 0.2


class RerankerParms(BaseDoc):
    top_n: PositiveInt = 1


class RAGASParams(BaseDoc):
    questions: List[TextDoc]
    answers: List[TextDoc]
    docs: List[TextDoc]
    ground_truths: List[TextDoc]


class RAGASScores(BaseDoc):
    answer_relevancy: float
    faithfulness: float
    context_recallL: float
    context_precision: float


class GraphDoc(BaseDoc):
    text: str
    strtype: Optional[str] = Field(
        description="type of input query, can be 'query', 'cypher', 'rag'",
        default="query",
    )
    max_new_tokens: Optional[int] = Field(default=1024)
    rag_index_name: Optional[str] = Field(default="rag")
    rag_node_label: Optional[str] = Field(default="Task")
    rag_text_node_properties: Optional[list] = Field(default=["name", "description", "status"])
    rag_embedding_node_property: Optional[str] = Field(default="embedding")


class LVMDoc(BaseDoc):
    image: Union[str, List[str]]
    prompt: str
    max_new_tokens: conint(ge=0, le=1024) = 512
    top_k: int = 10
    top_p: float = 0.95
    typical_p: float = 0.95
    temperature: float = 0.01
    repetition_penalty: float = 1.03
    stream: bool = False


class LVMVideoDoc(BaseDoc):
    video_url: str
    chunk_start: float
    chunk_duration: float
    prompt: str
    max_new_tokens: conint(ge=0, le=1024) = 512


class SDInputs(BaseDoc):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: int = 42
    negative_prompt: Optional[Union[str, List[str]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    lora_weight_name_or_path: Optional[str] = None


class SDImg2ImgInputs(BaseDoc):
    image: str
    prompt: Union[str, List[str]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    strength: float = 0.8
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: int = 1
    seed: int = 42
    lora_weight_name_or_path: Optional[str] = None


class SDOutputs(BaseDoc):
    images: list


class ImagePath(BaseDoc):
    image_path: str


class ImagesPath(BaseDoc):
    images_path: List[ImagePath]


class VideoPath(BaseDoc):
    video_path: str


class PrevQuestionDetails(BaseDoc):
    question: str
    answer: str


class PromptTemplateInput(BaseDoc):
    data: Dict[str, Any]
    conversation_history: Optional[List[PrevQuestionDetails]] = None
    conversation_history_parse_type: str = "naive"
    system_prompt_template: Optional[str] = None
    user_prompt_template: Optional[str] = None


class TranslationInput(BaseDoc):
    text: str
    target_language: str
