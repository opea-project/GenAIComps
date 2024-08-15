# RAG Agent with DocGrader

This strategy is specifically designed to improve answer quality over conventional RAG.
This agent strategy includes steps listed below:

1. RagAgent
   decide if this query need to get extra help

   - Yes: Goto 'Retriever'
   - No: Complete the query with Final answer

2. Retriever:

   - Get relative Info from tools, Goto 'DocumentGrader'

3. DocumentGrader
   Judge retrieved info relevance based on query

   - Yes: Go to TextGenerator
   - No: Go back to agent to rewrite query.

4. TextGenerator
   - Generate an answer based on query and last retrieved context.
   - After generation, go to END.

Note:

- The max number of retrieves is set at 3.
- You can specify a small `recursion_limit` to stop early or a big `recursion_limit` to fully use the 3 retrieves.
- The TextGenerator only looks at the last retrieved docs.

![Agentic Rag Workflow](https://blog.langchain.dev/content/images/size/w1000/2024/02/image-16.png)
