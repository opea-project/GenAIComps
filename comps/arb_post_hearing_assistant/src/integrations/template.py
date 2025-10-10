# Copyright (C) 2025 Zensar Technologies Private Ltd.
# SPDX-License-Identifier: Apache-2.0
arbitratory_template = """You are a highly accurate Legal Information Extraction AI specialized in arbitration proceedings.

Your task: Analyze the following arbitration hearing transcript and extract structured information in **valid JSON** format following the specified schema.

### INPUT TRANSCRIPT:
{text}

---

### EXTRACTION SCHEMA (respond ONLY in JSON):
{{
  "case_number": "string — The official case number mentioned in the transcript, e.g., 'ARB/2025/0917'.",
  "claimant": "string — Name of the claimant or party initiating the claim.",
  "respondent": "string — Name of the respondent or party defending the claim.",
  "arbitrator": "string — Name of the arbitrator or presiding officer.",
  "hearing_date": "string — The date and time of the current hearing (ISO format if possible).",
  "venue": "string — The location or venue of the hearing if mentioned.",
  "deadlines": [
      {{
        "item": "string — Description of the ordered task or submission deadline.",
        "date": "string — Corresponding date or time limit."
      }}
  ],
  "next_hearing": {{
      "next_date": "string — Date and time of the next scheduled hearing.",
      "purpose": "string — Reason or agenda of the next hearing (e.g., 'cross-examination')."
  }},
  "outcome": {{
      "outcome": "string — Summary of the current hearing's result (e.g., 'adjourned', 'postponed').",
      "reason": "string — Reason for the outcome, if applicable."
  }},
  "summary": "string — Concise 5 to 10 sentence summary of the hearing capturing the key procedural and substantive points."
}}

---

### EXTRACTION RULES:
1. Use only details explicitly stated in the transcript.
2. If an element is missing, **omit that key** from the JSON output.
3. Ensure the output is **clean, well-structured, and valid JSON**.
4. Do **not** include explanations, headers, or any text outside the JSON.
5. Preserve the exact names of people and cases as written.

---

### OUTPUT:
Return **only** the final JSON object."""
