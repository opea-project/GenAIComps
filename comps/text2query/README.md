# 🛢 Text-to-Query Microservice

A text2query microservice is a specialized, independent service designed to translate natural language queries into structured query languages. Its primary function is to act as an interpreter, allowing users to ask questions in plain human language and receive a formal query in return, which can then be executed against a Relational or Graph database. This service bridges the gap between human communication and machine-readable database commands.

## 🛠️ Features

- **Implement SQL Query based on input text**: Transform user-provided natural language into SQL queries, subsequently executing them to retrieve data from SQL databases.
- **Implement Cypher Query based on input text**: Transform user-provided natural language into Cypher queries, subsequently executing them to retrieve data from Neo4j Graph database.

## ⚙️ Supported Implementations

The Text2Query Microservice supports multiple implementation options to suit different databases. Each implementation includes its own configuration and setup instructions:

| Implementation     | Description                                                     | Supported Hardware | Documentation                  |
| ------------------ | --------------------------------------------------------------- | ------------------ | ------------------------------ |
| **Text-to-SQL**    | Transforming user-provided natural language into SQL queries    | Xeon, Gaudi        | [README](src/README_sql.md)    |
| **Text-to-Cypher** | Transforming user-provided natural language into Cypher queries | Gaudi              | [README](src/README_cypher.md) |
