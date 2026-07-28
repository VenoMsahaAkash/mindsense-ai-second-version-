# MindSense AI 🧠💙

> **An AI-powered Mental Health Assistant using RAG, Hybrid Classifier, Google Gemini 2.5 Flash, and Flask**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange.svg)](https://ai.google.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple.svg)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

MindSense AI is a production-ready, research-quality mental health assistant that combines:

- **Retrieval-Augmented Generation (RAG)** over a curated knowledge base (CBT, DBT, WHO, APA guidelines)
- **Hybrid Mental Health Classifier** (7-class: Normal, Stress, Depression, Anxiety, Suicidal, Bipolar, Personality Disorder)
- **Google Gemini 2.5 Flash** as the core LLM for empathetic response generation
- **Multi-Agent Architecture** with specialized agents for intent detection, risk assessment, deep analysis, validation, and summarization
- **Crisis Safety Protocol** with automatic escalation and resource referral

---

## 🏗️ Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Classifier  │  │ Intent Agent │  │   Risk Agent     │  │
│  │   Agent      │  │              │  │ (Crisis Detect)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌───────────────────────────────────┐   │
│  │  Analyzer    │  │        RAG Pipeline                │   │
│  │   Agent      │  │  FAISS → Reranker → PromptBuilder  │   │
│  └──────────────┘  └───────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Therapist   │  │  Validator   │  │  Summary Agent   │  │
│  │  Agent (LLM) │  │   Agent      │  │  (Memory)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
 Flask API → Chat UI
```

### Knowledge Base Pipeline
```
PDFs/TXT/MD → PyMuPDF Extract → RecursiveCharacterTextSplitter
    → SentenceTransformer Embeddings → FAISS Index
    → Hybrid Retrieval (Dense + BM25) → CrossEncoder Reranking
    → Prompt Injection → Gemini Response
```

---

## 🚀 Quick Start

### 1. Clone & Navigate

```bash
cd D:\MindSense-AI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First install may take 5–10 minutes (downloads PyTorch, transformers, FAISS).

### 4. Configure Environment

```bash
copy .env.example .env
```

Open `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

> Get your free API key at: https://aistudio.google.com/app/apikey

### 5. Add Knowledge Base Files *(Optional but recommended)*

Place PDF, TXT, or Markdown files into the appropriate subdirectories:

```
knowledge/
  CBT/          ← CBT therapy materials
  DBT/          ← DBT therapy materials
  WHO/          ← WHO mental health guidelines
  APA/          ← APA resources
  Coping/       ← Coping strategy guides
  Crisis/       ← Crisis intervention resources
  TherapistExamples/  ← Example therapy dialogues
```

### 6. Build the FAISS Knowledge Index

```bash
python rag/build_index.py
```

> Skip this step if you have no knowledge files — the app works without RAG (Gemini-only mode).

### 7. Run the Application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model to use |
| `GEMINI_TEMPERATURE` | `0.7` | Response creativity (0–1) |
| `FLASK_DEBUG` | `False` | Enable Flask debug mode |
| `FLASK_PORT` | `5000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` for embeddings |

---

## 📁 Project Structure

```
MindSense-AI/
├── app.py                    # Flask web application
├── config.py                 # Centralized configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container build
├── .env.example              # Environment template
│
├── model/
│   ├── classifier/           # Pre-trained classifier artifacts
│   ├── embedding_model/      # SentenceTransformer cache
│   ├── faiss/                # FAISS index & metadata
│   └── llm/
│       └── gemini_client.py  # Gemini API client
│
├── knowledge/                # Knowledge base documents
│   ├── CBT/ DBT/ WHO/ APA/
│   ├── Coping/ Crisis/ TherapistExamples/
│
├── rag/
│   ├── build_index.py        # Knowledge ingestion pipeline
│   ├── retriever.py          # Hybrid dense+sparse retrieval
│   ├── reranker.py           # Cross-encoder reranking
│   ├── embeddings.py         # Embedding wrapper
│   └── prompt_builder.py     # LLM prompt assembly
│
├── agents/
│   ├── orchestrator.py       # Pipeline coordinator
│   ├── classifier_agent.py   # Mental health classification
│   ├── intent_agent.py       # User intent detection
│   ├── risk_agent.py         # Crisis risk assessment
│   ├── analyzer_agent.py     # NLP emotion/theme analysis
│   ├── therapist_agent.py    # Gemini response generation
│   ├── validator_agent.py    # Response quality scoring
│   └── summary_agent.py      # Session summarization
│
├── prompts/
│   ├── system_prompt.py      # Core system instructions
│   ├── therapist_prompt.py   # Condition-specific guidance
│   ├── crisis_prompt.py      # Emergency override prompt
│   └── validator_prompt.py   # Self-evaluation prompt
│
├── memory/
│   ├── conversation_memory.py # Turn-by-turn history
│   ├── session_memory.py      # Session-level context
│   └── user_memory.py         # Cross-session persistence
│
├── evaluation/
│   ├── metrics.py             # BLEU, ROUGE, empathy, safety
│   ├── evaluate_classifier.py # Classifier benchmark
│   ├── evaluate_responses.py  # Response quality eval
│   └── benchmark.py           # Full research benchmark
│
├── utils/
│   ├── logger.py              # Structured JSON logging
│   ├── preprocessing.py       # Text cleaning/chunking
│   ├── helpers.py             # Shared utility functions
│   └── response_utils.py      # Response formatting
│
├── templates/
│   └── index.html             # Chat UI
│
└── static/
    ├── css/style.css          # Custom dark-mode CSS
    └── js/chat.js             # Chat interface logic
```

---

## 🔬 Research Contributions

This system addresses three key research challenges:

1. **Hybrid Retrieval for Mental Health**: Combines dense semantic search (FAISS/MiniLM) with sparse keyword matching for superior recall over clinical terminology.

2. **Multi-Agent Safety Architecture**: A dedicated RiskAgent + crisis protocol ensures immediate safety escalation, addressing a critical gap in LLM-based mental health systems.

3. **Automated Response Validation**: A 6-dimensional quality scoring system with automatic regeneration ensures consistent empathy, safety, and groundedness standards.

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t mindsense-ai .

# Run container
docker run -p 5000:5000 --env-file .env mindsense-ai
```

---

## 🧪 Evaluation

```bash
# Evaluate response quality
python evaluation/evaluate_responses.py

# Evaluate classifier (requires test dataset)
python evaluation/evaluate_classifier.py --dataset datasets/test.csv

# Full research benchmark
python evaluation/benchmark.py
```

---

## 🛡️ Ethics & Disclaimer

MindSense AI is a **research prototype** and must NOT be used as a replacement for professional mental health care. The system:

- Does **not** provide clinical diagnoses
- Does **not** prescribe medications
- Always recommends professional help for serious concerns
- Includes crisis resource referrals for high-risk interactions

**If you or someone you know is in crisis, please call a crisis helpline immediately.**


*Built with ❤️ for mental health research and engineering.*
