---
title: Scam Phishing Detector India
emoji: 🛡️
sdk: streamlit
app_file: app.py
---

# Indian SMS/WhatsApp Scam Detector

## What it does
This is an educational project that detects scam and phishing messages commonly seen in India via SMS and WhatsApp. It analyzes a message (in English, Hindi, or Hinglish) and provides a Risk Score (0-100), a Risk Level (Low/Medium/High), and a plain-English explanation. The system uses a rule-based Linguistic Agent and an LLM Verdict Agent augmented with factual evidence retrieved from a Knowledge Base (RAG).

## Architecture
```
[User Message]
      │
      ├─> 1. Linguistic Agent (Rules, Regex) ──> (Flags, Score)
      │
      ├─> 2a. Evidence Retrieval Agent (RAG) ──> (Top-k Facts/Scam Templates)
      │
      └─> 2b. Verdict Agent (Groq LLM) <── (Combined Context)
            │
            v
   [Risk Score, Level, Explanation, Advice]
```

## File-by-file explanation
- `app.py`: The Streamlit web interface for users to enter messages and view results.
- `linguistic.py`: Module 1 containing the rule-based Linguistic Agent that flags urgency, impersonation, etc.
- `rag.py`: Module 2a containing the FAISS vector index builder and Evidence Retrieval Agent using Sentence Transformers.
- `verdict.py`: Module 2b containing the Verdict Agent that builds the prompt and queries the Groq LLM (with fallback).
- `pipeline.py`: The main function `analyze(message)` that chains the three agents together.
- `evaluate.py`: Module 3 script to evaluate the pipeline and baselines on a test dataset.
- `data/kb_scam_messages.csv`: Knowledge base of synthetic scam message templates.
- `data/kb_trusted_info.csv`: Knowledge base of verified, safe-to-state facts from banks and government sources.
- `data/eval_sample.csv`: A small labeled dataset (text, label) for running evaluation out of the box.

## Run locally
1. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Copy the example environment file: `cp .env.example .env`
4. Add your Groq API key to `.env` as `GROQ_API_KEY=your_actual_key`
5. Run the app: `streamlit run app.py`

## Deploy on Hugging Face Spaces
1. Create a new "Streamlit" Space on Hugging Face.
2. Push this repository to the space.
3. In the Space Settings > Variables and secrets, add your `GROQ_API_KEY` as a Secret.
4. (Optional) Add `GROQ_MODEL` as a Variable if you want to use a different model.

## Alternative: Streamlit Community Cloud
If deploying on Streamlit Community Cloud, add the `GROQ_API_KEY` under the app Secrets settings in the dashboard. The code supports reading from `st.secrets` as a fallback.

## Run evaluation
To evaluate the pipeline against a TF-IDF baseline and a plain LLM baseline:
```bash
python evaluate.py --data data/eval_sample.csv --limit 100
```
*Note: Ensure your test dataset does not contain messages that exist in the knowledge base CSVs to avoid data leakage.*

## Where to put datasets
Put your datasets inside the `data/` directory. For evaluation, use `--data data/your_dataset.csv`. The script supports standard CSVs with `text` and `label` (0=legit, 1=scam) columns, or the UCI SMS Spam Collection format.

## Known limitations
- The Linguistic Agent relies on strict keywords, which sophisticated scams might evade.
- RAG retrieval uses a small embedding model; performance depends heavily on the quality and comprehensiveness of the knowledge base CSVs.
- This is an educational tool and should not be used as a definitive substitute for official verification.
