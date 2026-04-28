# Multimodal Voice Agent 

## Overview

This project is a **voice-enabled Retrieval-Augmented Generation (RAG) AI agent** built for the acedemic institute.
The system ingests content from **Sunmarke School’s website** and allows users to ask questions via **voice or text**.
The same query is processed simultaneously by **three different LLMs** and the responses are presented side by side in both **text and voice** format.

### Models Used

* **Gemini** (Google)
* **Kimi** (Moonshot AI)
* **DeepSeek**

The agent is strictly constrained to answer only from the ingested Sunmarke website content, with graceful fallback for out-of-scope questions.

---

## Key Features

* Website scraping and ingestion using LangChain
* Vector-based semantic search using FAISS
* Real-time browser-based voice recording
* Speech-to-text using Deepgram
* Retrieval-Augmented Generation (RAG)
* Parallel inference with 3 LLMs
* Text-to-speech output for all responses
* 3-column side-by-side comparison UI
* Secure API key handling via environment variables


## Architecture Overview

```
User (Voice / Text)
        |
        v
Speech-to-Text (Deepgram)
        |
        v
Shared User Query
        |
        v
Vector Retrieval (FAISS + Google Embeddings)
        |
        v
Prompt + Retrieved Context
        |
        +-------------------+-------------------+-------------------+
        |                   |                   |
        v                   v                   v
   Gemini (Google)      Kimi (Moonshot)     DeepSeek
        |                   |                   |
        v                   v                   v
   Text Response       Text Response        Text Response
        |                   |                   |
        +--------- Text-to-Speech (Edge TTS) ----+
                          |
                          v
                    Audio Playback
```

---

## Architecture Diagram

<img width="514" height="1345" alt="diagram-export-1-24-2026-7_17_04-PM" src="https://github.com/user-attachments/assets/6e3e3324-1524-4805-8558-6caf974b7f6b" />

## Tech Stack

### Backend / AI

* Python
* LangChain
* FAISS (Vector Store)
* Google Generative AI Embeddings
* OpenRouter (LLM routing)
* Deepgram (Speech-to-Text)
* Edge-TTS (Text-to-Speech)

### Frontend

* Streamlit
* streamlit-mic-recorder

### Deployment

* Streamlit community cloud

---

## Project Structure

```
.
├── ingest.py              # Website scraping & vector DB creation
├── app.py                 # Streamlit voice agent application
├── faiss_index/           # Generated vector store
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd virtuans-voice-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file using `.env.example` as reference and add:

```
USER_AGENT=myagent
OPENROUTER_API_KEY=
GOOGLE_API_KEY = 
OPENROUTER_BASE_URL=
DEEPGRAM_API_KEY=
```

No paid API keys are required.

---

## Running the Application

### Step 1: Ingest Website Data

This scrapes Sunmarke’s website and builds the vector database.

```bash
python ingest.py
```

### Step 2: Start the Streamlit App

```bash
streamlit run app.py
```

Open the provided local URL in your browser.

---

## Usage

1. Click **Record** and ask a question about Sunmarke School
   (admissions, fees, curriculum, facilities, etc.)
2. The query is transcribed once.
3. The same query is sent to all three models.
4. Responses are displayed side by side with:

   * Text output
   * Audio playback
   * Independent loading and error handling

Out-of-scope questions are handled gracefully with a polite explanation.

---

## Prompting & RAG Strategy

* Chunk size: `1000` characters
* Overlap: `200` characters
* Embeddings: `models/text-embedding-004` (Google)
* Top-k retrieval: `k = 3`
* Strict instruction to answer **only from retrieved context**
* Dynamic fallback messaging for non-Sunmarke questions

---

## Error Handling & Edge Cases

* Independent model failure handling
* Missing vector store detection
* Speech-to-text failure fallback
* Audio generation safety checks
* Environment variable validation at startup

---

## Estimated Cost (Free Tier)

Approximate cost per 1,000 queries:

* LLMs: Free tier via OpenRouter
* Embeddings: Free tier (Google)
* Speech-to-Text: Free Deepgram tier
* TTS: Local Edge TTS

**Estimated cost: $0.00 for assessment usage**

---

## Assumptions & Limitations

* Answers are limited strictly to scraped Sunmarke pages
* Website updates require re-running ingestion
* Mobile UI is functional but not fully optimized
* Real-time streaming STT is not used (pre-recorded chunks)

---

## Evaluation Coverage Checklist

* RAG pipeline implemented
* All 3 LLMs integrated and working
* Single voice input → 3 model outputs
* Voice output for all responses
* Clean UI with 3-column layout
* Secure API key handling
* Deployed-ready Streamlit app
* Clear documentation and architecture


## Author

**AI Engineer – Virtuans**

---
