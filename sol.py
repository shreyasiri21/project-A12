# app.py
import streamlit as st
import pandas as pd
from transformers import pipeline

# ---------------------------
# Local Hugging Face Model (no API)
# ---------------------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

nlp_model = load_model()

# ---------------------------
# Helper functions
# ---------------------------
def load_transactions(file):
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.lower()
        def fix_amount(r):
            amt, t = r["amount"], r["type"]
            if "debit" in t and amt > 0:
                return -abs(amt)
            if "credit" in t and amt < 0:
                return abs(amt)
            return amt
        df["amount"] = df.apply(fix_amount, axis=1)
    return df

def compute_summary(df):
    if df is None or "amount" not in df.columns:
        return None
    income = float(df[df["amount"] > 0]["amount"].sum())
    expense = float(-df[df["amount"] < 0]["amount"].sum())
    savings = round(income - expense, 2)
    rate = round((savings / income * 100) if income > 0 else 0.0, 2)
    top = {}
    if "description" in df.columns:
        neg = df[df["amount"] < 0]
        grouped = neg.groupby("description")["amount"].sum().sort_values()
        for desc, val in grouped.head(5).abs().items():
            top[desc] = round(abs(val), 2)
    return {"income": income, "expense": expense, "savings": savings, "rate": rate, "top": top}

def build_prompt(user_question, summary):
    system = (
        "You are a friendly personal finance helper for students. "
        "Explain in very simple words. Use short sentences. "
        "Give 2-3 simple tips. Do not use jargon."
    )
    data = ""
    if summary:
        data += f"\n\nUser's summary:\nIncome ₹{summary['income']}, Expense ₹{summary['expense']}, Savings ₹{summary['savings']}, Rate {summary['rate']}%"
        if summary.get("top"):
            data += "\nTop spending:\n"
            for k, v in summary["top"].items():
                data += f"- {k}: ₹{v}\n"
    return f"{system}{data}\n\nQuestion: {user_question}\nAnswer:"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Personal Finance Chatbot (Local)", layout="wide")
st.title("💬 Personal Finance Chatbot ")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = load_transactions(uploaded_file)
    if df is not None:
        st.subheader("📄 Transactions")
        st.dataframe(df.head(200))
        summary = compute_summary(df)
        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Income (₹)", f"{summary['income']}")
            c2.metric("Expense (₹)", f"{summary['expense']}")
            c3.metric("Savings (₹)", f"{summary['savings']}")
            c4.metric("Rate", f"{summary['rate']}%")
            if summary.get("top"):
                st.subheader("Top Spending")
                st.write(pd.DataFrame(list(summary["top"].items()), columns=["Item", "Amount"]))
else:
    df, summary = None, None
    st.info("Upload your CSV with columns: date, description, amount, type.")

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Chat")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_msg = st.chat_input("Ask something like 'How to save money?'")
if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})
    with st.spinner("Thinking..."):
        prompt = build_prompt(user_msg, summary)
        reply = nlp_model(prompt, max_new_tokens=200)[0]['generated_text']
    st.session_state.history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)