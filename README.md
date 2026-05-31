# 🤖 DoChat — Chat with Your Documents Locally

A private RAG (Retrieval-Augmented Generation) system that lets you upload PDF documents and query them using natural language — powered entirely by local LLMs. No data leaves your machine.

## Features

- 📄 Upload multiple PDF documents via drag & drop
- 🔍 Semantic search using vector embeddings
- 🧠 Local LLM inference via Ollama (no API keys required)
- 💬 Conversation memory within sessions
- 🔒 100% private — all processing happens on your machine
- 🐳 Docker-ready for easy deployment

## Tech Stack

| Layer | Technology |
|---|---|
| Interface | Streamlit |
| RAG Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | nomic-embed-text (via Ollama) |
| LLM | llama3.2 (via Ollama) |
| PDF Processing | PyPDF |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- llama3.2 and nomic-embed-text models pulled

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aerobalderas/dochat.git
cd dochat
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull required Ollama models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

5. Run the app:
```bash
streamlit run app/main.py
```

## Project Structure

```
dochat/
├── app/
│   ├── main.py        # Streamlit interface
│   ├── rag.py         # PDF processing and vector store
│   └── memory.py      # Conversation memory
├── uploads/           # Temporary PDF storage
├── vectorstore/       # ChromaDB vector database
├── requirements.txt
├── Dockerfile
└── README.md
```

## How It Works

1. Upload a PDF document through the sidebar
2. DoChat chunks the document and generates embeddings locally
3. Ask questions in natural language
4. The system retrieves relevant chunks and generates answers using a local LLM