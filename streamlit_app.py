# streamlit_app.py


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import nltk
import os
import shutil
from nltk import word_tokenize, pos_tag
import re


import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# --- Gemini API Key ---
genai.configure(api_key="AIzaSyBBxbeH81SEWus594hftEH-QiiBLnx5BuQ")

def extract_proper_nouns(text):
    tokens = word_tokenize(str(text))
    tagged = pos_tag(tokens)
    return {word for word, tag in tagged if tag in ('NNP', 'NNPS')}

# --- Row Matcher ---
def row_matcher(df, question):
    q_keywords = extract_proper_nouns(question)
    matched_rows = []
    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row.values if pd.notna(x))
        row_keywords = extract_proper_nouns(row_text)
        if any(qk.lower() in rk.lower() or rk.lower() in qk.lower() for qk in q_keywords for rk in row_keywords):
            matched_rows.append(row)
    return pd.DataFrame(matched_rows)

# --- Gemini Answer Generator ---
def generate_answer_only(question, context_df):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a clinical trials assistant. Use the context below to answer the question.
If the answer is not found, say \"I am not sure about this answer, please check the database.\"
Be factual and detailed. Ensure presentation is in tabular format.

Context:
{context_df.to_string(index=False)}

Question: {question}

Answer:"""
    return model.generate_content(prompt).text

# --- Gemini Plot Code Generator ---
def generate_plot_code_from_answer(context_str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a Python data visualization assistant. Given the following clinical trial summary, generate Python code using matplotlib or seaborn to visualize the most relevant metrics.
Ensure the code includes clear x axis ,y axis labels which are Descriptive and Accurate and Relavant, an informative title which is accurate, and uses the best-fit chart type (bar, pie, line, etc.) Make the graph detailed. The graph should be very relavant to the question.

{context_str}

Python code only:"""
    return model.generate_content(prompt).text

# --- Extract Code Block ---
def extract_code_block(text):
    match = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
    if not match:
        match = re.search(r"```(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


st.set_page_config(page_title="Clinical Trial QA + Viz", layout="wide")
st.title("🔬 Clinical Trial QA with Gemini + Visualization")

uploaded_file = st.file_uploader("📁 Upload Clinical Trial Excel File", type=["xlsx"])

questions = [
    "Please compare ORR, CR, PR, mPFS, and mOS of M14TIL regimen with that of checkmate067's nivolumab + ipilimumab?",
    "How do ORR, CR, PFS, OS, and Gr ≥3 TRAEs compare between CHECKMATE-511 and CHECKMATE-067 regimens?",
    "How many patients died during the CHECKMATE-511 trial, of these how many are treatment related?",
    "What was the regimen studied in IMspire150 trial? How does it compare against COMBI-d regimen in terms of efficacy and safety?",
    "Which studies have Ph3 outcomes, how do they compare in key parameters - ORR, CR, PFS, OS, DOR, Gr ≥3 TRAEs or TEAEs?",
    "How the DREAMseq is different from CHECKMATE-067?"
]

# --- Session State Initialization and Reset ---
if "prev_question" not in st.session_state:
    st.session_state.prev_question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

selected_question = st.selectbox("❓ Select a Question", questions)

# Reset answer if question changed
if selected_question != st.session_state.prev_question:
    st.session_state.answer = ""
    st.session_state.prev_question = selected_question

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    #st.subheader("📄 Matched Rows")
    matched_df = row_matcher(df, selected_question)
    #st.dataframe(matched_df)

    st.subheader("🧠 Gemini Q/A Answer")

    # Display existing answer
    if st.session_state.answer:
        st.markdown(st.session_state.answer)

    # Button to generate answer
    if st.button("Generate Answer"):
        answer_text = generate_answer_only(selected_question, matched_df)
        st.session_state.answer = answer_text
        st.markdown(answer_text)

    # Button to generate graph
    if st.session_state.answer.strip() != "":
        st.subheader("📊 Graph Based on Answer")
        if st.button("Generate Graph from Answer"):
            full_context = f"""Matched Rows:\n{matched_df.to_string(index=False)}\n\nQuestion:\n{selected_question}\n\nAnswer:\n{st.session_state.answer}"""
            plot_code_response = generate_plot_code_from_answer(full_context)

            st.subheader("📈 Executed Plot")
            code_block = extract_code_block(plot_code_response)
            if code_block:
                try:
                    exec(code_block)
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"❌ Error executing generated code: {e}")
            else:
                st.warning("⚠️ No valid Python code block found.")
else:
    st.info("Upload an Excel file to begin.")