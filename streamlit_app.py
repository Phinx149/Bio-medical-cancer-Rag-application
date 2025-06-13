import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import nltk
import os
import re
from nltk import word_tokenize, pos_tag

# --- 1. Streamlit UI Setup (MUST BE FIRST) ---
st.set_page_config(page_title="Clinical Trial QA + Viz", layout="wide")
st.title("🔬 Clinical Trial QA with Gemini + Visualization")

# --- 2. Setup NLTK Data Directory ---
# The path to your COMMITTED nltk_data folder on Streamlit Cloud.
# Streamlit Cloud deploys your repo to /mount/src/your-repo-name/
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.environ['NLTK_DATA'] = '/mount/src/bio-medical-cancer-rag-application/nltk_data'
# Set the NLTK_DATA environment variable. This is CRUCIAL for NLTK to find the data.
#os.environ['NLTK_DATA'] = nltk_data_path

# Add our custom data path to NLTK's search path (optional, but good for redundancy)
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# --- 3. ***REMOVED ALL NLTK Data Download Logic*** ---
# The data is now expected to be available in the committed 'nltk_data' folder.
# The 'download_nltk_resource' function and its calls are GONE.

# --- Gemini API Key ---
genai.configure(api_key="AIzaSyBBxbeH81SEWus594hftEH-QiiBLnx5BuQ")

# --- Proper noun extractor ---
def extract_proper_nouns(text):
    # NLTK should now find 'punkt' and 'averaged_perceptron_tagger'
    # because NLTK_DATA environment variable points to the committed data.
    tokens = word_tokenize(str(text))
    tagged = pos_tag(tokens)
    return {word for word, tag in tagged if tag in ('NNP', 'NNPS')}

# --- Match rows based on keywords ---
def row_matcher(df, question):
    q_keywords = extract_proper_nouns(question)
    matched_rows = []
    for _, row in df.iterrows():
        row_text = " ".join(str(x) for x in row.values if pd.notna(x))
        row_keywords = extract_proper_nouns(row_text)
        if any(qk.lower() in rk.lower() or rk.lower() in qk.lower() for qk in q_keywords for rk in row_keywords):
            matched_rows.append(row)
    return pd.DataFrame(matched_rows)

# --- Answer using Gemini ---
def generate_answer_only(question, context_df):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a clinical trials assistant. Use the context below to answer the question.
If the answer is not found, say "I am not sure about this answer, please check the database."
Be factual and detailed. Ensure presentation is in tabular format.

Context:
{context_df.to_string(index=False)}

Question: {question}

Answer:"""
    return model.generate_content(prompt).text

# --- Generate Plot Code from Gemini ---
def generate_plot_code_from_answer(context_str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""You are a Python data visualization assistant. Given the following clinical trial summary, generate Python code using matplotlib or seaborn to visualize the most relevant metrics.
Ensure the code includes clear axis labels, an informative title, and uses the best-fit chart type (bar, pie, line, etc.)

{context_str}

Python code only:"""
    return model.generate_content(prompt).text

# --- Extract code block ---
def extract_code_block(text):
    match = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
    if not match:
        match = re.search(r"```(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


uploaded_file = st.file_uploader("📁 Upload Clinical Trial Excel File", type=["xlsx"])

# Load Excel from upload or GitHub fallback
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Using uploaded dataset.")
else:
    GITHUB_RAW_URL = "https://github.com/Phinx149/Bio-medical-cancer-Rag-application/raw/refs/heads/main/Sample%20Data.xlsx"
    df = pd.read_excel(GITHUB_RAW_URL)
    st.info("ℹ️ Using default dataset from GitHub.")

# --- Clean columns ---
df.columns = [col.strip() for col in df.columns]

# --- Questions ---
questions = [
    "Please compare ORR, CR, PR, mPFS, and mOS of M14TIL regimen with that of checkmate067's nivolumab + ipilimumab?",
    "How do ORR, CR, PFS, OS, and Gr ≥3 TRAEs compare between CHECKMATE-511 and CHECKMATE-067 regimens?",
    "How many patients died during the CHECKMATE-511 trial, of these how many are treatment related?",
    "What was the regimen studied in IMspire150 trial? How does it compare against COMBI-d regimen in terms of efficacy and safety?",
    "Which studies have Ph3 outcomes, how do they compare in key parameters - ORR, CR, PFS, OS, DOR, Gr ≥3 TRAEs or TEAEs?",
    "How the DREAMseq is different from CHECKMATE-067?"
]

# --- Session State ---
if "prev_question" not in st.session_state:
    st.session_state.prev_question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

# --- UI: Select Question ---
selected_question = st.selectbox("❓ Select a Question", questions)

# Reset answer if question changed
if selected_question != st.session_state.prev_question:
    st.session_state.answer = ""
    st.session_state.prev_question = selected_question

# --- Match rows ---
matched_df = row_matcher(df, selected_question)

# --- Answer section ---
st.subheader("🧠 Gemini Q/A Answer")
if st.session_state.answer:
    st.markdown(st.session_state.answer)

if st.button("Generate Answer"):
    answer_text = generate_answer_only(selected_question, matched_df)
    st.session_state.answer = answer_text
    st.markdown(answer_text)

# --- Graph section ---
if st.session_state.answer.strip() != "":
    st.subheader("📊 Graph Based on Answer")
    if st.button("Generate Graph from Answer"):
        full_context = f"""Matched Rows:\n{matched_df.to_string(index=False)}\n\nQuestion:\n{selected_question}\n\nAnswer:\n{st.session_state.answer}"""
        plot_code_response = generate_plot_code_from_answer(full_context)
        code_block = extract_code_block(plot_code_response)
        if code_block:
            try:
                exec(code_block)
                st.pyplot(plt.gcf())
                plt.close('all')
            except Exception as e:
                st.error(f"❌ Error executing generated code: {e}")
        else:
            st.warning("⚠️ No valid Python code block found.")