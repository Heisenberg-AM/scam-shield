from linguistic import analyze_linguistics, clean_text
from rag import retrieve_evidence
from verdict import verdict_agent

def analyze(message):
    """
    Chains the three agents (Linguistic, Evidence Retrieval, Verdict) to analyze a message.
    Returns a unified dictionary with all results.
    """
    # 1. Clean the text (useful for debugging, though linguistic agent cleans internally)
    cleaned = clean_text(message)

    # 2. Module 1: Linguistic Analysis (Rule-based)
    linguistic_result = analyze_linguistics(message)

    # 3. Module 2a: Retrieve Evidence (RAG)
    evidence = retrieve_evidence(message)

    # 4. Module 2b: Verdict Agent (LLM + Fallback)
    verdict = verdict_agent(message, linguistic_result, evidence)

    # Combine results
    return {
        "cleaned_text": cleaned,
        "linguistic_result": linguistic_result,
        "evidence": evidence,
        "risk_score": verdict.get("risk_score", 0),
        "level": verdict.get("level", "Low"),
        "explanation": verdict.get("explanation", ""),
        "reasons": verdict.get("reasons", []),
        "advice": verdict.get("advice", ""),
        "evidence_used": verdict.get("evidence_used", []),
        "fallback": verdict.get("fallback", False)
    }
