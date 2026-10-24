import argparse
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

from pipeline import analyze
from verdict import plain_llm_baseline

# Constants
SCAM_LEVELS = ["Medium", "High"]
API_DELAY_SECONDS = 1

def evaluate_system(y_true, y_pred, system_name):
    """
    Computes and prints evaluation metrics for a system.
    Returns the metrics dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {system_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "System": system_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate scam detection systems.")
    parser.add_argument("--data", type=str, default="data/eval_sample.csv", help="Path to evaluation dataset")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of test rows for LLM evaluation")
    args = parser.parse_args()

    # 1. Load Data
    try:
        if args.data.endswith(".tsv") or "spam.csv" in args.data:
            # Handle UCI SMS Spam Collection format roughly
            df = pd.read_csv(args.data, sep='\t' if args.data.endswith('.tsv') else ',', encoding='latin-1')
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df.rename(columns={'v1': 'label', 'v2': 'text'})
                df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        else:
            df = pd.read_csv(args.data)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Ensure columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
         print("Dataset must have 'text' and 'label' columns.")
         return

    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)

    print(f"Loaded {len(df)} total samples.")

    # 2. Train/Test Split (Stratified to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
    )

    # Apply limit for LLM API cost/rate limits
    X_test_eval = X_test.head(args.limit)
    y_test_eval = y_test.head(args.limit)

    print(f"Evaluating on {len(X_test_eval)} test samples.")

    metrics_results = []

    # --- System A: TF-IDF + LinearSVC Baseline ---
    print("\nTraining TF-IDF + LinearSVC baseline on train set...")
    pipeline_tfidf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LinearSVC(random_state=42))
    ])
    pipeline_tfidf.fit(X_train, y_train)
    y_pred_tfidf = pipeline_tfidf.predict(X_test_eval)
    metrics_results.append(evaluate_system(y_test_eval, y_pred_tfidf, "TF-IDF + LinearSVC"))

    # --- System B & C: Evaluate LLM Systems ---
    y_pred_plain_llm = []
    y_pred_proposed = []
    proposed_outputs = []

    api_key_available = os.environ.get("GROQ_API_KEY") is not None
    if not api_key_available:
        print("\nWARNING: GROQ_API_KEY not found. LLM pipelines will use fallback.")

    print("\nRunning Plain LLM and Proposed Pipeline on test set...")
    for idx, (text, true_label) in enumerate(zip(X_test_eval, y_test_eval)):
        print(f"Processing {idx+1}/{len(X_test_eval)}...")

        # System B: Plain LLM
        def call_plain_llm():
            res = plain_llm_baseline(text)
            return 1 if res.get("level") in SCAM_LEVELS else 0

        try:
            pred_b = call_plain_llm()
        except Exception:
            time.sleep(API_DELAY_SECONDS) # Wait and retry once
            try:
                pred_b = call_plain_llm()
            except Exception:
                pred_b = 0 # Default to legit on total failure

        y_pred_plain_llm.append(pred_b)

        time.sleep(API_DELAY_SECONDS)

        # System C: Proposed Pipeline
        def call_proposed():
            return analyze(text)

        try:
            res_c = call_proposed()
        except Exception:
            time.sleep(API_DELAY_SECONDS)
            try:
                res_c = call_proposed()
            except Exception:
                res_c = {"level": "Low", "risk_score": 0, "explanation": "Failed"}

        pred_c = 1 if res_c.get("level") in SCAM_LEVELS else 0
        y_pred_proposed.append(pred_c)

        proposed_outputs.append({
            "message": text,
            "true_label": true_label,
            "predicted_level": res_c.get("level"),
            "risk_score": res_c.get("risk_score"),
            "explanation": res_c.get("explanation")
        })

        time.sleep(API_DELAY_SECONDS)

    metrics_results.append(evaluate_system(y_test_eval, y_pred_plain_llm, "Plain LLM Baseline"))
    metrics_results.append(evaluate_system(y_test_eval, y_pred_proposed, "Proposed RAG Pipeline"))

    # Save Results
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(metrics_results).to_csv("results/metrics.csv", index=False)
    pd.DataFrame(proposed_outputs).to_csv("results/pipeline_outputs.csv", index=False)
    print("\nSaved metrics to results/metrics.csv and detailed outputs to results/pipeline_outputs.csv")

if __name__ == "__main__":
    main()
