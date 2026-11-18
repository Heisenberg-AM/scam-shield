import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
LEVEL_LOW_MAX = 34
LEVEL_MEDIUM_MAX = 64
DEFAULT_MODEL = "gpt-oss-20b"

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
        fallback_result["explanation"] = "Could not reach LLM. Based purely on rule-based keyword detection. (Reason: GROQ_API_KEY is not set or empty)"
        return fallback_result

    client = Groq(api_key=api_key)
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    # Format evidence
    evidence_text = ""
    for i, ev in enumerate(evidence):
        evidence_text += f"[{i+1}] Source: {ev['source_type']}\nText: {ev['text']}\n\n"

    prompt = f"""You are a specialized cybersecurity and scam-detection analyst for Indian SMS and WhatsApp messages.
Analyze the target message below by evaluating the rule-based linguistic findings alongside the retrieved threat intelligence.

Target Message:
"{message}"

Linguistic Engine Output:
- Preliminary Score: {linguistic_result['score']}/100
- Detected Threat Categories: {json.dumps(linguistic_result['flags'])}
- Matched Keywords/Entities: {json.dumps(linguistic_result['matched'])}

Retrieved Similar Threat Patterns (Vector DB):
{evidence_text}

---
### Analysis Guidelines:
1. **Linguistic & Structural Flag Verification**:
   - Inspect the message for temporal urgency or threats (e.g., "within 12 hours", "tonight at 9:30 PM", "account blocked"). If present, treat it as high risk even if no immediate monetary fee is demanded.
   - Inspect links and domains: Check if legitimate Indian entities (e.g., India Post, SBI, BESCOM/State Electricity, TRAI) are paired with non-official, third-party, or typo-squatted domains (e.g., `.top`, `-update.com`, link shorteners like `bit.ly`/`tinyurl`).
   - Reverse-urgency / Call-to-action: Identify reverse-authorization traps where inaction defaults to a loss (e.g., "SMS BLOCK to number" or "auto-debit scheduled").
   - Unrealistic work/reward claims (e.g., liking videos, part-time daily payouts via WhatsApp links).

2. **Synthesis with Retrieved Evidence**:
   - High similarity (≥ 0.60): Leverage pattern matches directly to confirm intent.
   - Low similarity (< 0.50): If retrieved items are irrelevant or legitimate bank templates, **do not downgrade** an obviously malicious structure (fake domains, urgency, harvesting). Rely on the linguistic red flags and domain structure.

3. **Risk Scoring Calibration**:
   - 0-25: Clearly legitimate notifications, standard transactional updates (OTP, balance debit/credit from official shortcodes).
   - 26-55: Ambiguous, promotional spam, or unverified claims without suspicious URLs or credential requests.
   - 56-85: Obvious phishing, credential/KYC harvesting, fake disconnection threats, or lookalike tracking domains.
   - 86-100: Critical active attacks (APK drops, digital arrest threats, direct financial extortion, SIM swap triggers).

---
Respond strictly in valid JSON format matching this schema:
{{
  "risk_score": <integer 0-100>,
  "risk_level": "<LOW | MEDIUM | HIGH | CRITICAL>",
  "detected_linguistic_flags": ["<flag 1>", "<flag 2>"],
  "explanation": "<2-3 clear sentences highlighting the deceptive mechanism and why the linguistic flags matter>",
  "reasons": [
    "<short concrete point about impersonation/domain>",
    "<short concrete point about urgency/coercion>"
  ],
  "evidence_used": [<list of integer IDs from retrieved items relied on, or empty list [] if judged purely on structural flags>],
  "advice": "<single concise, actionable safety directive>"
}}
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
        fallback_result["explanation"] = f"Could not reach LLM. Based purely on rule-based keyword detection. (API Error: {str(e)})"
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
