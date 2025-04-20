# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from collections import namedtuple
from string import Template
from typing import Any, Dict, List, Optional, Union

from langchain_community.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

from comps import CustomLogger

logger = CustomLogger("opea_text2cypher_cypher_utils")
Relationship = namedtuple("Relationship", ["entity1", "relationship", "entity2"])

cypher_cleanup = """
 MATCH (n) DETACH DELETE n
"""

cypher_insert = """
 LOAD CSV WITH HEADERS FROM 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQCEUxVlMZwwI2sn2T1aulBrRzJYVpsM9no8AEsYOOklCDTljoUIBHItGnqmAez62wwLpbvKMr7YoHI/pub?gid=0&single=true&output=csv' AS rows
 MERGE (d:disease {name:rows.Disease})
 MERGE (dt:diet {name:rows.Diet})
 //SET dt.Diet = rows.Diet
 MERGE (d)-[:HOME_REMEDY]->(dt)

 MERGE (m:medication {name:rows.Medication})
 //SET m.Medication = rows.Medication
 MERGE (d)-[:TREATMENT]->(m)

 MERGE (s:symptom {name:rows.Symptom})
 //SET s.Symptoms = rows.Symptom
 MERGE (d)-[:MANIFESTATION]->(s)

 MERGE (p:precaution {name:rows.Precaution})
 //SET p.precaution = rows.Precaution
 MERGE (d)-[:PREVENTION]->(p)

 //MATCH (n:Description) RETURN count(n) AS count
"""


# in the original string, change the first character of the substring into lower case
def replace_with_lowercase(s, sub):
    index = s.lower().find(sub)
    if index < 0 or index >= len(s):
        raise ValueError("Index out of range")
    # Replace the character at the index with its lowercase version
    return s[:index] + s[index].lower() + s[index + 1 :]


def parse_relationships(input_str):
    # Split by comma to get each relationship
    substrings = input_str.split(",")

    relationships = []

    for substring in substrings:
        # Remove unnecessary characters and split by spaces
        parts = (
            substring.replace("(", "")
            .replace(")", "")
            .replace(">", "")
            .replace("[", "")
            .replace("]", "")
            .replace(":", "")
            .split("-")
        )

        entity1 = parts[0].strip()
        relationship = parts[1].strip()
        entity2 = parts[2].strip()

        # Create a named tuple and add it to the list
        relationships.append(Relationship(entity1, relationship, entity2))

    return relationships


def swap(cypher_string, relations):
    # Regular expression pattern to match substrings
    pattern = r"\(([^()]*)\)"

    # Find all matches
    matches = re.findall(pattern, cypher_string)
    new_string = cypher_string
    # Check if there are exactly two matches
    if len(matches) == 2:
        first = matches[0]
        second = matches[1]
        for rel in relations:
            e1 = rel.entity1.lower()
            r = rel.relationship.lower()
            e2 = rel.entity2.lower()
            if e1 in first.lower() and e2 in second.lower():
                new_string = replace_with_lowercase(new_string, e1)
                new_string = replace_with_lowercase(new_string, e2)
                break
            elif e1 in second.lower() and e2 in first.lower():
                new_string = new_string.replace(f"({second})", "tmp", 1)
                new_string = new_string.replace(f"({first})", f"({second})", 1)
                new_string = new_string.replace("tmp", f"({first})", 1)
                new_string = replace_with_lowercase(new_string, e1)
                new_string = replace_with_lowercase(new_string, e2)
                break
            elif e1 in first.lower() and r in second.lower():
                new_string = new_string.replace(f"{second[2:]}", f"{e2}", 1)
                new_string = replace_with_lowercase(new_string, e1)
                new_string = replace_with_lowercase(new_string, e2)
                break
            elif e1 in second.lower() and r in first.lower():
                new_string = new_string.replace(f"({second})", "tmp", 1)
                new_string = new_string.replace(f"({first})", f"({second})", 1)
                new_string = new_string.replace("tmp", f"({first})", 1)
                # now replace r with e2
                new_string = new_string.replace(f"{first[2:]}", f"{e2}", 1)
                new_string = replace_with_lowercase(new_string, e1)
                new_string = replace_with_lowercase(new_string, e2)
                break
    return new_string


def prepare_chat_template(question):
    template = Template(
        """
Generate Cypher statement to query a graph database to answer "$question"
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Use only the user's question to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Cypher output:
    """
    )
    temp_str = template.substitute(question=question)
    return temp_str


def construct_schema(
    structured_schema: Dict[str, Any],
    include_types: List[str],
    exclude_types: List[str],
) -> str:
    """Filter the schema based on included or excluded types."""

    def filter_func(x: str) -> bool:
        return x in include_types if include_types else x not in exclude_types

    filtered_schema: Dict[str, Any] = {
        "node_props": {k: v for k, v in structured_schema.get("node_props", {}).items() if filter_func(k)},
        "rel_props": {k: v for k, v in structured_schema.get("rel_props", {}).items() if filter_func(k)},
        "relationships": [
            r
            for r in structured_schema.get("relationships", [])
            if all(filter_func(r[t]) for t in ["start", "end", "type"])
        ],
    }

    # Format node properties
    formatted_node_props = []
    for label, properties in filtered_schema["node_props"].items():
        props_str = ", ".join([f"{prop['property']}: {prop['type']}" for prop in properties])
        formatted_node_props.append(f"{label} {{{props_str}}}")

    # Format relationship properties
    formatted_rel_props = []
    for rel_type, properties in filtered_schema["rel_props"].items():
        props_str = ", ".join([f"{prop['property']}: {prop['type']}" for prop in properties])
        formatted_rel_props.append(f"{rel_type} {{{props_str}}}")

    # Format relationships
    formatted_rels = [f"(:{el['start']})-[:{el['type']}]->(:{el['end']})" for el in filtered_schema["relationships"]]

    return "\n".join(
        [
            "Node properties are the following:",
            ",".join(formatted_node_props),
            "Relationship properties are the following:",
            ",".join(formatted_rel_props),
            "The relationships are the following:",
            ",".join(formatted_rels),
        ]
    )


class CypherQueryCorrectorExt(CypherQueryCorrector):
    """Extension of CypherQueryCorrector. CypherQueryCorrector corrects
    relationship direction in generated Cypher statements.

    CypherQueryCorrectorExt can perform additional checks based on a schema.
    For example,
         1. remove any extra information from the cypher statement.
         2. any user-defined and/or application-specific transformation.
    """

    def __init__(self, schemas: List[Schema], schema_str: str):
        self.schemas = schemas
        self.schema_str = schema_str

    def correct_query(self, query: str) -> str:
        logger.info(f"[ correct_query ] query: {query}")
        start_index = query.find("MATCH ")
        if start_index == -1:
            raise ValueError("Generated cypher does not contain `MATCH `.")
        tmp1 = query[start_index:]
        pattern = r"-\[.*?\]->"
        replacement = "-[INTERACT_WITH]->"
        tmp2 = re.sub(pattern, replacement, tmp1)

        rel_index = self.schema_str.find("The relationships are the following:\n")
        rel_string = self.schema_str[rel_index + len("The relationships are the following:\n") :]
        relations = parse_relationships(rel_string)
        query = swap(tmp2, relations)
        logger.info(f"[ correct_query ] corrected query: {query}")
        return query

    def __call__(self, query: str) -> str:
        """Correct the query to make it valid.

        If
        Args:
            query: cypher query
        """
        return self.correct_query(query)
