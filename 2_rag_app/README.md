# 🏛️ LegisAI — Congressional Bills RAG Chatbot

**Project 2 of 2** | [← See the Data Pipeline (Project 1)](../1_data_pipeline/)

A Retrieval-Augmented Generation (RAG) chatbot for searching and understanding U.S. Congressional bills from the 118th Congress. It combines semantic vector search with real-time LLaMA-3 summarization and XGBoost-powered passage probability predictions.

---

##  Architecture

```
User Query
    │
    ▼
SentenceTransformer          all-MiniLM-L6-v2
    │  (embed query)
    ▼
ChromaDB                     cosine similarity search
    │  (retrieve top-5 bill chunks)
    ▼
Groq LLaMA-3.3-70B           1. Generates grounded 3-sentence global answer
    │                        2. Concurrently generates 1-sentence summaries for each bill
    ▼
XGBoost Classifier           Predicts passage probability + extracts signals per bill
    │
    ▼
Streamlit UI                 Displays clean, light-themed interactive dashboard
```

---

## Setup & Usage

```bash
# 1. Activate your virtual environment
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
# Edit the .env file and add your Groq API key (https://console.groq.com)
```

### Step 1 — Build the vector index (first time only)
```bash
python rag_pipeline.py
```
*This reads the cleaned data, chunks it, embeds with `all-MiniLM-L6-v2`, and indexes into ChromaDB. Takes ~5 minutes.*

### Step 2 — Run the app
```bash
streamlit run app.py
```

---

##  Features

| Feature | Details |
|---|---|
| **Semantic Search** | Natural language queries across 2,200+ embedded congressional bills. |
| **Global Context Answer** | Groq (LLaMA-3) reads all retrieved snippets and synthesizes a global answer to your query. |
| **Dynamic Bill Summaries** | Uses concurrent API calls to generate plain-English, 1-sentence summaries for every retrieved bill on the fly. |
| **Probability & Signals** | Integrates an XGBoost model to display the percentage chance a bill will pass, alongside human-readable signals (e.g., "Senate Bill", "Bypassed Committee"). |
| **Interactive UI** | Modern, clean light-themed UI with status indicator left-borders (green/red) and visual probability progress bars. |

---

##  Deployment (Hugging Face Spaces)
This application is fully configured to be deployed on Hugging Face Spaces. The `config.py` uses dynamic path resolution, allowing you to drag and drop `app.py`, `config.py`, the `chroma_db/` directory, and your `.pkl/.csv` files directly into a Hugging Face Streamlit Space for instant, free hosting.

 **Live Demo:**
*https://huggingface.co/spaces/shruthirathod/LegisAI-Bills-Tracker*
---

##  Tech Stack
- **Retrieval**: ChromaDB, Hugging Face `sentence-transformers`
- **LLM**: Groq API (`llama-3.3-70b-versatile`)
- **Concurrent Processing**: `concurrent.futures.ThreadPoolExecutor`
- **ML**: XGBoost (Passage Predictor)
- **UI**: Streamlit + Custom HTML/CSS
