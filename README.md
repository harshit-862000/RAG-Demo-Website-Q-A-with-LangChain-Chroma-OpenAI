# RAG Demo — Website Q&A with LangChain & OpenAI

A Retrieval-Augmented Generation (RAG) pipeline in Python that enables natural language Q&A over live website content. It scrapes URLs, splits text into chunks, stores embeddings in a Chroma vector database, and uses an OpenAI LLM via LangChain to generate concise, context-grounded answers.

---

## 🚀 Features

- Scrapes and loads content from any public URL
- Splits documents into chunks for efficient retrieval
- Stores and queries semantic embeddings using Chroma
- Generates accurate, context-grounded answers via OpenAI LLM
- Built and tested on Google Colab

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | Orchestration & retrieval chains |
| Chroma | Vector store for embeddings |
| OpenAI | Embeddings & LLM completions |
| Unstructured | HTML content extraction from URLs |
| Sentence Transformers | Embedding utilities |
| Google Colab | Development environment |

---

## ⚙️ Installation
```bash
pip install langchain==0.3.25 \
langchain-community==0.3.23 \
langchain-openai==0.3.11 \
python-dotenv==1.0.1 \
langchain-experimental==0.3.4 \
sentence-transformers==3.4.1 \
langchain-chroma==0.2.4 \
langchainhub==0.1.21 \
unstructured==0.17.2
```

---

## 🔑 Environment Variables

Set the following in your environment or Colab secrets:
```
OPENAI_API_KEY=your_openai_api_key
BASE_URL=your_openai_base_url
```

---

## 📖 How It Works

1. **Ingest** — URLs are scraped using `UnstructuredURLLoader`
2. **Chunk** — Content is split using `RecursiveCharacterTextSplitter` (chunk size: 1000)
3. **Embed** — Chunks are embedded using OpenAI embeddings and stored in Chroma
4. **Retrieve** — Top-3 relevant chunks are fetched via similarity search
5. **Generate** — An OpenAI LLM generates a concise answer using retrieved context

---

## 💡 Example Query
```python
response = rag_chain.invoke({"input": "What is the admission schedule for LKG class?"})
print(response["answer"])
```

---

## 📁 Project Structure
```
RAG_demo.ipynb   # Main notebook with full pipeline
README.md        # Project documentation
```

---

## ⚠️ Known Issue

The final chain invocation may raise a `BadRequestError` if the base `OpenAI` (completion) LLM is used with a chat-style prompt. Fix by replacing it with `ChatOpenAI`:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")
```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
