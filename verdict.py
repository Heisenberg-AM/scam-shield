import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
LEVEL_LOW_MAX = 34
LEVEL_MEDIUM_MAX = 64
DEFAULT_MODEL = "llama-3.1-8b-instant"

def get_level(score):
    """
    Converts a 0-100 risk score into a Low/Medium/High level.
    """
    if score <= LEVEL_LOW_MAX:
        return "Low"
    elif score <= LEVEL_MEDIUM_MAX:
        return "Medium"
    else:
        return "High"

def verdict_agent(message, linguistic_result, evidence):
    """
    Builds a prompt from the message, linguistic findings, and RAG evidence,
    then queries the Groq API to get a risk score and explanation.
    Provides a fallback to the linguistic score if the API call fails or API key is missing.
    """
    # Build fallback response first
    score = linguistic_result["score"]
    level = get_level(score)
    reasons = [f"Triggered {flag}" for flag, is_set in linguistic_result["flags"].items() if is_set]
    if not reasons:
         reasons = ["No suspicious linguistic signals detected."]

    fallback_result = {
        "risk_score": score,
        "level": level,
        "explanation": "Could not reach LLM. Based purely on rule-based keyword detection.",
        "reasons": reasons,
        "evidence_used": [],
        "advice": "Be cautious and verify with official sources.",
        "fallback": True
    }

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # If running without API key, use fallback silently (UI will show notice)
        return fallback_result

    client = Groq(api_key=api_key)
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    # Format evidence
    evidence_text = ""
    for i, ev in enumerate(evidence):
        evidence_text += f"[{i+1}] Source: {ev['source_type']}\nText: {ev['text']}\n\n"

    prompt = f"""You are a scam-detection analyst for Indian SMS/WhatsApp.
Analyze this message: "{message}"

Linguistic findings: Risk score {linguistic_result['score']}/100.
Flags: {json.dumps(linguistic_result['flags'])}
Matched keywords: {json.dumps(linguistic_result['matched'])}

Retrieved Evidence:
{evidence_text}

Use ONLY the provided evidence and linguistic findings. If the evidence is weak or contradicts, say so. Legitimate messages exist, so do not label everything as a scam.
Respond ONLY with JSON format:
{{"risk_score": integer between 0-100, "explanation": "2-4 simple sentences", "reasons": ["short bullet 1", "short bullet 2"], "evidence_used": [numbers of the evidence items you relied on], "advice": "one short safety tip"}}
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        response_json = json.loads(completion.choices[0].message.content)

        # Ensure risk_score is present and is an integer
        if "risk_score" not in response_json:
             response_json["risk_score"] = score
        else:
             response_json["risk_score"] = int(response_json["risk_score"])

        response_json["level"] = get_level(response_json["risk_score"])
        response_json["fallback"] = False
        return response_json

    except Exception as e:
        print(f"LLM API Error: {e}")
        return fallback_result


def plain_llm_baseline(message):
    """
    Zero-shot Groq call with NO linguistic findings and NO retrieval.
    Used only by evaluate.py.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"risk_score": 0, "level": "Low", "fallback": True}

    client = Groq(api_key=api_key)
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    prompt = f"""You are a scam-detection analyst for Indian SMS/WhatsApp.
Analyze this message: "{message}"

Respond ONLY with JSON format:
{{"risk_score": integer between 0-100, "explanation": "2-4 simple sentences", "reasons": ["short bullet 1"], "evidence_used": [], "advice": "one short safety tip"}}
"""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        response_json = json.loads(completion.choices[0].message.content)
        if "risk_score" not in response_json:
             response_json["risk_score"] = 0
        else:
             response_json["risk_score"] = int(response_json["risk_score"])

        response_json["level"] = get_level(response_json["risk_score"])
        response_json["fallback"] = False
        return response_json

    except Exception as e:
        print(f"Baseline LLM API Error: {e}")
        return {"risk_score": 0, "level": "Low", "fallback": True}
