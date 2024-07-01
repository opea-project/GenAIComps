# LanceDB

LanceDB is an embedded vector database for AI applications. It is open source and distributed with an Apache-2.0 license.

LanceDB datasets are persisted to disk and can be shared between Node.js and Python.

## Setup

```bash
npm install -S vectordb
```

## Usage

### Create a new index from texts

```javascript
import { LanceDB } from "@langchain/community/vectorstores/lancedb";
import { OpenAIEmbeddings } from "@langchain/openai";
import { connect } from "vectordb";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import os from "node:os";

export const run = async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "lancedb-"));
  const db = await connect(dir);
  const table = await db.createTable("vectors", [{ vector: Array(1536), text: "sample", id: 1 }]);

  const vectorStore = await LanceDB.fromTexts(
    ["Hello world", "Bye bye", "hello nice world"],
    [{ id: 2 }, { id: 1 }, { id: 3 }],
    new OpenAIEmbeddings(),
    { table },
  );

  const resultOne = await vectorStore.similaritySearch("hello world", 1);
  console.log(resultOne);
  // [ Document { pageContent: 'hello nice world', metadata: { id: 3 } } ]
};
```

API Reference:

- `LanceDB` from `@langchain/community/vectorstores/lancedb`
- `OpenAIEmbeddings` from `@langchain/openai`

### Create a new index from a loader

```javascript
import { LanceDB } from "@langchain/community/vectorstores/lancedb";
import { OpenAIEmbeddings } from "@langchain/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { connect } from "vectordb";

// Create docs with a loader
const loader = new TextLoader("src/document_loaders/example_data/example.txt");
const docs = await loader.load();

export const run = async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "lancedb-"));
  const db = await connect(dir);
  const table = await db.createTable("vectors", [{ vector: Array(1536), text: "sample", source: "a" }]);

  const vectorStore = await LanceDB.fromDocuments(docs, new OpenAIEmbeddings(), { table });

  const resultOne = await vectorStore.similaritySearch("hello world", 1);
  console.log(resultOne);

  // [
  //   Document {
  //     pageContent: 'Foo\nBar\nBaz\n\n',
  //     metadata: { source: 'src/document_loaders/example_data/example.txt' }
  //   }
  // ]
};
```

API Reference:

- `LanceDB` from `@langchain/community/vectorstores/lancedb`
- `OpenAIEmbeddings` from `@langchain/openai`
- `TextLoader` from `langchain/document_loaders/fs/text`

### Open an existing dataset

```javascript
import { LanceDB } from "@langchain/community/vectorstores/lancedb";
import { OpenAIEmbeddings } from "@langchain/openai";
import { connect } from "vectordb";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import os from "node:os";

// You can open a LanceDB dataset created elsewhere, such as LangChain Python, by opening
// an existing table
export const run = async () => {
  const uri = await createdTestDb();
  const db = await connect(uri);
  const table = await db.openTable("vectors");

  const vectorStore = new LanceDB(new OpenAIEmbeddings(), { table });

  const resultOne = await vectorStore.similaritySearch("hello world", 1);
  console.log(resultOne);
  // [ Document { pageContent: 'Hello world', metadata: { id: 1 } } ]
};

async function createdTestDb(): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "lancedb-"));
  const db = await connect(dir);
  await db.createTable("vectors", [
    { vector: Array(1536), text: "Hello world", id: 1 },
    { vector: Array(1536), text: "Bye bye", id: 2 },
    { vector: Array(1536), text: "hello nice world", id: 3 },
  ]);
  return dir;
}
```

API Reference:

- `LanceDB` from `@langchain/community/vectorstores/lancedb`
- `OpenAIEmbeddings` from `@langchain/openai`
