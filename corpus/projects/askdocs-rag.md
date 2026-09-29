# askdocs-rag

Portfolio site card: Document Q&A with a retrieval and answer-quality evaluation harness — grounded, cited answers that abstain when the answer is not in the docs. Stack: Python · RAG · Chroma · LLM. Published site line: hit@k / MRR retrieval · LLM-as-judge for faithfulness · 19 tests.

From the public README: askdocs-rag ingests markdown, text, and PDF, chunks them (800 characters, 120 overlap), embeds locally with sentence-transformers/all-MiniLM-L6-v2, stores vectors in Chroma with cosine similarity, and answers from retrieved passages with citations. The sample corpus in that repo is fictional project documentation, not this biography. The eval harness reports retrieval hit@k and MRR separately from an LLM-as-judge, and it scores abstention on unanswerable questions. This note does not copy that sample run's scores.

Source: https://github.com/Ashishkosana/askdocs-rag

ask-ashish, the API behind this chat, reuses that shape for the portfolio corpus. It is a different repository: https://github.com/Ashishkosana/ask-ashish
