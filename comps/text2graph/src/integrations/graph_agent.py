# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import math
import os
import re
from typing import List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TripletExtractor:
    def triplet_extractor(self, text):
        triplets = []
        relation, subject, relation, object_ = "", "", "", ""
        text = text.strip()
        current = "x"
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = "t"
                if relation != "":
                    triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
                    relation = ""
                subject = ""
            elif token == "<subj>":
                current = "s"
                if relation != "":
                    triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
                object_ = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    object_ += " " + token
                elif current == "o":
                    relation += " " + token
        if subject != "" and relation != "" and object_ != "":
            triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
        return triplets


class TripletBuilder:
    def __init__(self):
        # Load model and tokenizer
        MODEL_NAME = os.environ.get("LLM_MODEL_ID", "Babelscape/rebel-large")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        ## Defines
        self.span_length = int(os.environ.get("SPAN_LENGTH", "1024"))
        self.overlap = int(os.environ.get("OVERLAP", "100"))
        self.model = model
        self.tokenizer = tokenizer

    async def cal_index_span(self, total_tokens, span_length, overlap):
        num_spans = math.ceil(total_tokens / span_length) + 1  # Calculate number of spans and assign to num_spans
        spans = []  # Initialize an empty list to store the spans
        start = 0
        for i in range(num_spans):  # Iterate using the calculated num_spans
            start = i * (span_length - overlap)
            end = min(start + span_length, total_tokens)  # Calculate end
            if end >= total_tokens:
                end = total_tokens
                start = end - span_length
            if span_length <= overlap:
                raise ValueError("Indexing is incorrect something is wrong")

            spans.append([start, end])  # Append the span to the list
        return spans

    async def gen_tokenize(self, text: str) -> List[str]:
        # print(f'entering tokenizer {text[:100]}')
        tensor_tokens = self.tokenizer([text], return_tensors="pt")
        # print(f'done entering tokenizer {tensor_tokens}')
        return tensor_tokens

    ## code
    async def extract_graph(self, text):
        # print(f'Entering graph extraction')
        tokenize_input = await self.gen_tokenize(text)
        total_tokens = len(tokenize_input["input_ids"][0])
        span_index_gen = await self.cal_index_span(total_tokens, self.span_length, self.overlap)
        tensor_ids = [torch.tensor(tokenize_input["input_ids"][0][start:end]) for start, end in span_index_gen]
        tensor_masks = [torch.tensor(tokenize_input["attention_mask"][0][start:end]) for start, end in span_index_gen]
        rearrange_inputs = {"input_ids": torch.stack(tensor_ids), "attention_mask": torch.stack(tensor_masks)}

        # generate relations
        MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
        num_return_sequences = 3
        gen_kwargs = {
            "max_length": MAX_LENGTH,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": num_return_sequences,
        }

        generated_tokens = self.model.generate(**rearrange_inputs, **gen_kwargs)

        # decode relations
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # create kb
        tripmgr = TripletManager()
        tripext = TripletExtractor()
        i = 0

        for sentence_pred in decoded_preds:
            current_span_index = i // num_return_sequences
            relations = tripext.triplet_extractor(sentence_pred)
            for relation in relations:
                tripmgr.add_relation(relation)
            i += 1
        return tripmgr


class TripletManager:
    def __init__(self):
        self.entities = {}  # { entity_title: {...} }
        self.relations = []  # [ head: entity_title, type: category, tail: entity_title]

    def are_relations_equal(self, relation1, relation2):
        """Check if two relations are equal."""
        head_match = relation1["head"] == relation2["head"]
        type_match = relation1["type"] == relation2["type"]
        tail_match = relation1["tail"] == relation2["tail"]
        all_match = head_match and type_match and tail_match
        return all_match

    def exists_relation(self, relation1):
        """Check if relation exists."""
        return any(self.are_relations_equal(relation1, relation2) for relation2 in self.relations)

    def merge_relations(self, relation2):
        """Merge two relations."""
        relation1 = [r for r in self.relations if self.are_relations_equal(relation2, r)][0]

    def exists_entity(self, entity_title):
        return entity_title in self.entities

    def add_entity(self, entity):
        """Check if entry exists and add if not."""
        if self.exists_entity(entity):  # Directly check if the entity exists
            return
        self.entities[entity] = {"title": entity}  # Create a dictionary for the entity
        return

    def add_relation(self, relation):
        """Add entry checking to see if it needs merge or create a new entry."""
        candidate_entities = [relation["head"], relation["tail"]]

        # manage new entities
        for entity in candidate_entities:
            self.add_entity(entity)

        # manage new relation
        if not self.exists_relation(relation):
            self.relations.append(relation)
        else:
            self.merge_relations(relation)

    def write_to_csv(self, WRITE_TO_CSV=False):
        """Saves the entities and relations to a CSV file."""
        struct_entity = {"entity": [], "details": []}
        struct_triplets = {"head": [], "type": [], "tail": []}

        # Instead of appending, build lists of entities and relations
        entity_data = []
        for entity in self.entities.items():
            entity_data.append(entity)

        relation_data = []
        for relation in self.relations:
            relation_data.append(relation)

        # Create DataFrames from the collected data
        df_entity = pd.DataFrame(entity_data, columns=["entity", "details"])
        df_relation = pd.DataFrame(relation_data)

        # Write to CSV if requested
        if WRITE_TO_CSV:
            df_entity.to_csv("entities.csv", index=True)
            df_relation.to_csv("relations.csv", index=True)
        return df_entity, df_relation
