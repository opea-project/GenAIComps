# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import re
import unicodedata

import easyocr
import fitz
import numpy as np
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from PIL import Image


def uni_pro(text):
    """Check if the character is ASCII or falls in the category of non-spacing marks."""
    normalized_text = unicodedata.normalize("NFKD", text)
    filtered_text = ""
    for char in normalized_text:
        if ord(char) < 128 or unicodedata.category(char) == "Mn":
            filtered_text += char
        elif "\u4E00" <= char <= "\u9FFF":
            filtered_text += char
        elif (
            "\u3400" <= char <= "\u4DBF"  # CJK Unified Ideographs Extension A
            or "\u20000" <= char <= "\u2A6DF"  # CJK Unified Ideographs Extension B
            or "\u2A700" <= char <= "\u2B73F"  # CJK Unified Ideographs Extension C
            or "\u2B740" <= char <= "\u2B81F"  # CJK Unified Ideographs Extension D
            or "\u2B820" <= char <= "\u2CEAF"  # CJK Unified Ideographs Extension E
            or "\uF900" <= char <= "\uFAFF"  # CJK Compatibility Ideographs
            or "\u2F800" <= char <= "\u2FA1F"
        ):
            filtered_text += char
    return filtered_text


def load_unstructured_data(input, table_strategy):
    """Load unstructured context."""
    tables = None
    if input.endswith("pdf"):
        text, tables = read_pdf(input, table_strategy)

    text = text.replace("\n", " ")
    text = text.replace("\n\n", " ")
    text = uni_pro(text)
    text = re.sub(r"\s+", " ", text)
    return text, tables


def read_pdf(pdf_path, table_strategy):
    """Read the pdf file."""
    doc = fitz.open(pdf_path)
    reader = easyocr.Reader(["en"], gpu=False)
    result = ""
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pagetext = page.get_text().strip()
        if pagetext:
            if pagetext.endswith("!") or pagetext.endswith("?") or pagetext.endswith("."):
                result = result + pagetext
            else:
                result = result + pagetext + "."
        if len(doc.get_page_images(i)) > 0:
            for img in doc.get_page_images(i):
                if img:
                    pageimg = ""
                    xref = img[0]
                    img_data = doc.extract_image(xref)
                    img_bytes = img_data["image"]
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    img = np.array(pil_image)
                    img_result = reader.readtext(img, paragraph=True, detail=0)
                    pageimg = pageimg + ", ".join(img_result).strip()
                    if pageimg.endswith("!") or pageimg.endswith("?") or pageimg.endswith("."):
                        pass
                    else:
                        pageimg = pageimg + "."
                result = result + pageimg
    tables_result = get_tables_result(pdf_path, table_strategy)
    return result, tables_result


def get_tables_result(pdf_path, table_strategy):
    """Extract tables information from pdf file."""
    if table_strategy == "fast":
        return None

    from unstructured.documents.elements import FigureCaption
    from unstructured.partition.pdf import partition_pdf

    # from intel_extension_for_transformers.neural_chat.models.model_utils import predict
    # from intel_extension_for_transformers.neural_chat.prompts.prompt import TABLESUMMARY_PROMPT

    tables_result = []
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
    )
    tables = [el for el in raw_pdf_elements if el.category == "Table"]
    for table in tables:
        table_coords = table.metadata.coordinates.points
        content = table.metadata.text_as_html
        table_page_number = table.metadata.page_number
        min_distance = float("inf")
        table_summary = None
        if table_strategy == "hq":
            for element in raw_pdf_elements:
                if isinstance(element, FigureCaption) or element.text.startswith("Tab"):
                    caption_page_number = element.metadata.page_number
                    caption_coords = element.metadata.coordinates.points
                    related, y_distance = get_relation(
                        table_coords, caption_coords, table_page_number, caption_page_number
                    )
                    if related:
                        if y_distance < min_distance:
                            min_distance = y_distance
                            table_summary = element.text
            if table_summary is None:
                parent_id = table.metadata.parent_id
                for element in raw_pdf_elements:
                    if element.id == parent_id:
                        table_summary = element.text
                        break
        elif table_strategy == "llm":
            # prompt = TABLESUMMARY_PROMPT.format(table_content=content)
            # params = {}
            # params["model_name"] = table_summary_model_name_or_path
            # params["prompt"] = prompt
            # params["temperature"] = 0.8
            # params["top_p"] = 0.9
            # params["top_k"] = 40
            # params["max_new_tokens"] = 1000
            # params["num_beams"] = 2
            # params["num_return_sequences"] = 2
            # params["use_cache"] = True
            # table_summary = predict(**params)
            table_summary = llm_generate(content)
            table_summary = table_summary[table_summary.find("### Generated Summary:\n") :]
            table_summary = re.sub("### Generated Summary:\n", "", table_summary)
        elif table_strategy == None:
            table_summary = None
        if table_summary is None:
            text = f"[Table: {content}]"
        else:
            text = f"|Table: [Summary: {table_summary}], [Content: {content}]|"
        tables_result.append([text, pdf_path])
    return tables_result


def llm_generate(content):
    llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
    llm = HuggingFaceEndpoint(
        endpoint_url=llm_endpoint,
        max_new_tokens=1000,
        top_k=40,
        top_p=0.9,
        temperature=0.8,
        streaming=False,
        num_beams=2,
        num_return_sequences=2,
        use_cache=True,
        timeout=600,
    )

    table_summary_template = """
    Task: Your task is to give a concise summary of the table. \
    The summary should cover the overall table structure and all detailed information of the table. \
    The table will be given in html format. Summarize the table below.
    ---
    ### Table:
    {table_content}
    ---
    ### Generated Summary:
    """

    prompt = PromptTemplate(template=table_summary_template, input_variables=["table_content"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.invoke(content)
    response = response["text"]
    print("response", response)
    return response


def get_relation(table_coords, caption_coords, table_page_number, caption_page_number, threshold=100):
    """Get the relation of a pair of table and caption."""
    same_page = table_page_number == caption_page_number
    x_overlap = (min(table_coords[2][0], caption_coords[2][0]) - max(table_coords[0][0], caption_coords[0][0])) > 0
    if table_coords[0][1] - caption_coords[1][1] >= 0:
        y_distance = table_coords[0][1] - caption_coords[1][1]
    elif caption_coords[0][1] - table_coords[1][1] >= 0:
        y_distance = caption_coords[0][1] - table_coords[1][1]
    else:
        y_distance = 0
    y_close = y_distance < threshold
    return same_page and x_overlap and y_close, y_distance


def get_chuck_data(content, max_length, min_length, input):
    """Process the context to make it maintain a suitable length for the generation."""
    sentences = re.split("(?<=[!.?])", content)

    paragraphs = []
    current_length = 0
    count = 0
    current_paragraph = ""
    for sub_sen in sentences:
        count += 1
        sentence_length = len(sub_sen)
        if current_length + sentence_length <= max_length:
            current_paragraph += sub_sen
            current_length += sentence_length
            if count == len(sentences) and len(current_paragraph.strip()) > min_length:
                paragraphs.append([current_paragraph.strip(), input])
        else:
            paragraphs.append([current_paragraph.strip(), input])
            current_paragraph = sub_sen
            current_length = sentence_length

    return paragraphs
