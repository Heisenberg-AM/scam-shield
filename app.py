import streamlit as st
import os
import pandas as pd
from pipeline import analyze
from rag import load_resources

# Setup Streamlit page
st.set_page_config(page_title="Scam & Phishing Detector", page_icon="🛡️")

# Initialize RAG index in memory
load_resources()

# Apply secrets from Streamlit Cloud to environment if missing
if not os.environ.get("GROQ_API_KEY"):
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

st.title("🛡️ Indian SMS/WhatsApp Scam Detector")
st.markdown("Analyze messages for phishing and scams using Linguistic Rules and RAG-Augmented LLM Analysis.")

# Example messages
EXAMPLES = {
    "UPI Collect Scam": "Dear user, you have a collect request of Rs 10,000 pending on PhonePe. Please enter your UPI PIN to receive the money immediately.",
    "KYC Link Scam": "Urgent: Your HDFC Bank account is suspended. Complete your KYC within 24 hours to reactivate. Visit http://hdfc.kyc-now.top",
    "Hinglish Prize Scam": "Inaam jeeta! Aapne Rs 5000 ka cashback jeeta hai. Paisa claim karne ke liye apna UPI PIN dalein.",
    "Job Scam": "Work from home! Earn Rs 5000 daily by liking YouTube videos. Contact us on WhatsApp for details.",
    "Legit Bank Alert": "Dear Customer, Rs 1000 debited from A/c XX1234 via UPI. Available Bal: Rs 5000."
}

# Example selector
selected_example = st.selectbox("Try an example message:", [""] + list(EXAMPLES.keys()))
default_text = EXAMPLES.get(selected_example, "")

# Text Input
message = st.text_area("Or paste a message here:", value=default_text, height=150)

# Analyze Button
if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        with st.spinner("Analyzing message..."):
            result = analyze(message)

        # UI Rendering for Results
        st.divider()

        # 1. Risk Badge and Score
        level = result.get("level", "Low")
        score = result.get("risk_score", 0)

        color_map = {"Low": "green", "Medium": "orange", "High": "red"}
        badge_color = color_map.get(level, "gray")

        st.markdown(f"### Risk Level: <span style='color:{badge_color}; font-weight:bold;'>{level.upper()}</span>", unsafe_allow_html=True)
        st.progress(score / 100, text=f"Risk Score: {score}/100")

        # Warning if API failed
        if result.get("fallback"):
             st.warning("⚠️ LLM API unavailable (check API Key). Showing rule-based linguistic analysis only.")

        # 2. Explanation and Advice
        st.markdown("#### Explanation")
        st.write(result.get("explanation", "No explanation provided."))

        if result.get("reasons"):
            st.markdown("#### Why?")
            for reason in result.get("reasons", []):
                st.markdown(f"- {reason}")

        if result.get("advice"):
             st.info(f"💡 **Advice:** {result.get('advice')}")

        # 3. Linguistic Signals (Expandable)
        with st.expander("🔍 Linguistic Signals"):
            ling_res = result.get("linguistic_result", {})
            st.write(f"**Base Linguistic Score:** {ling_res.get('score', 0)}")
            flags = ling_res.get("flags", {})
            matched = ling_res.get("matched", {})

            has_signals = False
            for flag, is_set in flags.items():
                if is_set:
                    has_signals = True
                    st.markdown(f"**{flag.replace('_', ' ').title()}** detected:")
                    st.write(f"Matched words: `{(', '.join(matched.get(flag, [])))}`")

            if not has_signals:
                st.write("No suspicious linguistic patterns detected.")

        # 4. Retrieved Evidence (Expandable)
        with st.expander("📚 Retrieved Evidence from Knowledge Base"):
            evidence = result.get("evidence", [])
            if evidence:
                for i, ev in enumerate(evidence):
                    st.markdown(f"**Item {i+1}** (Source: `{ev.get('source_type')}`, Similarity: {ev.get('similarity', 0):.2f})")
                    st.write(f"> {ev.get('text')}")
            else:
                st.write("No highly relevant evidence found in the knowledge base.")

st.divider()

# Optional: Show Evaluation Metrics if they exist
if os.path.exists("results/metrics.csv"):
    with st.expander("📊 Evaluation Results (Metrics)"):
        try:
            df_metrics = pd.read_csv("results/metrics.csv")
            st.dataframe(df_metrics)
        except Exception as e:
            st.error(f"Could not load metrics: {e}")

st.caption("Educational project. Not a substitute for official verification. Report fraud at cybercrime.gov.in or 1930.")
