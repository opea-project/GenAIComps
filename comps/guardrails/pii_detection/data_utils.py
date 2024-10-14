# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import json
import multiprocessing
import os
import re
import subprocess
import unicodedata
from urllib.parse import urlparse, urlunparse

import easyocr
import fitz
import numpy as np
import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from docx import Document as DDocument
from langchain_community.document_loaders import (
    UnstructuredImageLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredXMLLoader,
)
from PIL import Image


def load_pdf(pdf_path):
    """Load the pdf file."""
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
    return result


def load_html(html_path):
    """Load the html file."""
    with open(html_path, "r", encoding="utf-8") as file:
        html = file.read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(strip=True)
    return text


def load_txt(txt_path):
    """Load txt file."""
    with open(txt_path, "r") as file:
        text = file.read()
    return text


def load_doc(doc_path):
    """Load doc file."""
    txt_path = doc_path.replace(".doc", ".txt")
    try:
        with open(txt_path, "w") as outfile:
            subprocess.run(["antiword", doc_path], stdout=outfile, check=True)
    except:
        raise AssertionError(
            "antiword failed or not installed, if not installed,"
            + 'use "apt-get update && apt-get install -y antiword" to install it.'
        )
    text = load_txt(txt_path)
    os.remove(txt_path)
    return text


def load_docx(docx_path):
    """Load docx file."""
    doc = DDocument(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text


def load_pptx(pptx_path):
    """Load pptx file."""
    loader = UnstructuredPowerPointLoader(pptx_path)
    text = loader.load()[0].page_content
    return text


def load_md(md_path):
    """Load md file."""
    loader = UnstructuredMarkdownLoader(md_path)
    text = loader.load()[0].page_content
    return text


def load_xml(xml_path):
    """Load xml file."""
    loader = UnstructuredXMLLoader(xml_path)
    text = loader.load()[0].page_content
    return text


def load_json(json_path):
    """Load and process json file."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return json.dumps(data)


def load_yaml(yaml_path):
    """Load and process yaml file."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return yaml.dump(data)


def load_xlsx(input_path):
    """Load and process xlsx file."""
    df = pd.read_excel(input_path)
    return df.to_string()


def load_csv(input_path):
    """Load the csv file."""
    df = pd.read_csv(input_path)
    return df.to_string()


def load_image(image_path):
    """Load the image file."""
    loader = UnstructuredImageLoader(image_path)
    text = loader.load()[0].page_content
    return text


def load_svg(svg_path):
    """Load the svg file."""
    import cairosvg

    png_path = svg_path.replace(".svg", ".png")
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    text = load_image(png_path)
    os.remove(png_path)
    return text


def document_loader(doc_path):
    if doc_path.endswith(".pdf"):
        return load_pdf(doc_path)
    elif doc_path.endswith(".html"):
        return load_html(doc_path)
    elif doc_path.endswith(".txt"):
        return load_txt(doc_path)
    elif doc_path.endswith(".doc"):
        return load_doc(doc_path)
    elif doc_path.endswith(".docx"):
        return load_docx(doc_path)
    elif doc_path.endswith(".pptx") or doc_path.endswith(".ppt"):
        return load_pptx(doc_path)
    elif doc_path.endswith(".md"):
        return load_md(doc_path)
    elif doc_path.endswith(".xml"):
        return load_xml(doc_path)
    elif doc_path.endswith(".json") or doc_path.endswith(".jsonl"):
        return load_json(doc_path)
    elif doc_path.endswith(".yaml"):
        return load_yaml(doc_path)
    elif doc_path.endswith(".xlsx") or doc_path.endswith(".xls"):
        return load_xlsx(doc_path)
    elif doc_path.endswith(".csv"):
        return load_csv(doc_path)
    elif doc_path.endswith(".tiff"):
        return load_image(doc_path)
    elif doc_path.endswith(".svg"):
        return load_image(doc_path)
    else:
        raise NotImplementedError(
            "Current only support pdf, html, txt, doc, docx, pptx, ppt, md, xml"
            + ", json, jsonl, yaml, xlsx, xls, csv, tiff and svg format."
        )


class Crawler:
    def __init__(self, pool=None):
        if pool:
            assert isinstance(pool, (str, list, tuple)), "url pool should be str, list or tuple"
        self.pool = pool
        self.headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng, \
            */*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, \
            like Gecko) Chrome/113.0.0.0 Safari/537.36",
        }
        self.fetched_pool = set()

    def get_sublinks(self, soup):
        sublinks = []
        for links in soup.find_all("a"):
            sublinks.append(str(links.get("href")))
        return sublinks

    def get_hyperlink(self, soup, base_url):
        sublinks = []
        for links in soup.find_all("a"):
            link = str(links.get("href"))
            if link.startswith("#") or link is None or link == "None":
                continue
            suffix = link.split("/")[-1]
            if "." in suffix and suffix.split(".")[-1] not in ["html", "htmld"]:
                continue
            link_parse = urlparse(link)
            base_url_parse = urlparse(base_url)
            if link_parse.path == "":
                continue
            if link_parse.netloc != "":
                # keep crawler works in the same domain
                if link_parse.netloc != base_url_parse.netloc:
                    continue
                sublinks.append(link)
            else:
                sublinks.append(
                    urlunparse(
                        (
                            base_url_parse.scheme,
                            base_url_parse.netloc,
                            link_parse.path,
                            link_parse.params,
                            link_parse.query,
                            link_parse.fragment,
                        )
                    )
                )
        return sublinks

    def fetch(self, url, headers=None, max_times=5):
        if not headers:
            headers = self.headers
        while max_times:
            if not url.startswith("http") or not url.startswith("https"):
                url = "http://" + url
            print("start fetch %s...", url)
            try:
                response = requests.get(url, headers=headers, verify=True)
                if response.status_code != 200:
                    print("fail to fetch %s, response status code: %s", url, response.status_code)
                else:
                    return response
            except Exception as e:
                print("fail to fetch %s, caused by %s", url, e)
                raise Exception(e)
            max_times -= 1
        return None

    def process_work(self, sub_url, work):
        response = self.fetch(sub_url)
        if response is None:
            return []
        self.fetched_pool.add(sub_url)
        soup = self.parse(response.text)
        base_url = self.get_base_url(sub_url)
        sublinks = self.get_hyperlink(soup, base_url)
        if work:
            work(sub_url, soup)
        return sublinks

    def crawl(self, pool, work=None, max_depth=10, workers=10):
        url_pool = set()
        for url in pool:
            base_url = self.get_base_url(url)
            response = self.fetch(url)
            soup = self.parse(response.text)
            sublinks = self.get_hyperlink(soup, base_url)
            self.fetched_pool.add(url)
            url_pool.update(sublinks)
            depth = 0
            while len(url_pool) > 0 and depth < max_depth:
                print("current depth %s...", depth)
                mp = multiprocessing.Pool(processes=workers)
                results = []
                for sub_url in url_pool:
                    if sub_url not in self.fetched_pool:
                        results.append(mp.apply_async(self.process_work, (sub_url, work)))
                mp.close()
                mp.join()
                url_pool = set()
                for result in results:
                    sublinks = result.get()
                    url_pool.update(sublinks)
                depth += 1

    def parse(self, html_doc):
        soup = BeautifulSoup(html_doc, "lxml")
        return soup

    def download(self, url, file_name):
        print("download %s into %s...", url, file_name)
        try:
            r = requests.get(url, stream=True, headers=self.headers, verify=True)
            f = open(file_name, "wb")
            for chunk in r.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)
        except Exception as e:
            print("fail to download %s, caused by %s", url, e)

    def get_base_url(self, url):
        result = urlparse(url)
        return urlunparse((result.scheme, result.netloc, "", "", "", ""))

    def clean_text(self, text):
        text = text.strip().replace("\r", "\n")
        text = re.sub(" +", " ", text)
        text = re.sub("\n+", "\n", text)
        text = text.split("\n")
        return "\n".join([i for i in text if i and i != " "])


def uni_pro(text):
    """Check if the character is ASCII or falls in the category of non-spacing marks."""
    normalized_text = unicodedata.normalize("NFKD", text)
    filtered_text = ""
    for char in normalized_text:
        if ord(char) < 128 or unicodedata.category(char) == "Mn":
            filtered_text += char
    return filtered_text


def load_html_data(url):
    crawler = Crawler()
    res = crawler.fetch(url)
    if res is None:
        return None
    soup = crawler.parse(res.text)
    all_text = crawler.clean_text(soup.select_one("body").text)
    main_content = ""
    for element_name in ["main", "container"]:
        main_block = None
        if soup.select(f".{element_name}"):
            main_block = soup.select(f".{element_name}")
        elif soup.select(f"#{element_name}"):
            main_block = soup.select(f"#{element_name}")
        if main_block:
            for element in main_block:
                text = crawler.clean_text(element.text)
                if text not in main_content:
                    main_content += f"\n{text}"
            main_content = crawler.clean_text(main_content)
    main_content = all_text if main_content == "" else main_content
    main_content = main_content.replace("\n", "")
    main_content = main_content.replace("\n\n", "")
    main_content = uni_pro(main_content)
    main_content = re.sub(r"\s+", " ", main_content)

    return main_content


def parse_html(input):
    """Parse the uploaded file."""
    chucks = []
    for link in input:
        if re.match(r"^https?:/{2}\w.+$", link):
            content = load_html_data(link)
            if content is None:
                continue
            chuck = [[content.strip(), link]]
            chucks += chuck
        else:
            print("The given link/str {} cannot be parsed.".format(link))

    return chucks
