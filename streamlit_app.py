import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import nltk
import os
import re
from nltk import word_tokenize, pos_tag
import shutil



# --- Setup NLTK Data Directory ---
nltk_data_path = "/content/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
shutil.rmtree(os.path.join(nltk_data_path, 'tokenizers'), ignore_errors=True)
shutil.rmtree(os.path.join(nltk_data_path, 'taggers'), ignore_errors=True)
shutil.rmtree('/root/nltk_data/tokenizers', ignore_errors=True)
shutil.rmtree('/root/nltk_data/taggers', ignore_errors=True)
nltk.data.path = [nltk_data_path]
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
try:
    nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
except Exception as e:
    print(f"Warning: Could not download 'averaged_perceptron_tagger_eng'. Error: {e}")
try:
    nltk.download('punkt_tab', download_dir=nltk_data_path)
except Exception as e:
    print(f"Warning: Could not download 'punkt_tab'. Error: {e}")

# The rest of your code remains the same...
# --- Gemini API Key ---
genai.configure(api_key="AIzaSyBBxbeH81SEWus594hftEH-QiiBLnx5BuQ")

# --- Proper noun extractor ---
def extract_proper_nouns(text):
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

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Clinical Trial QA + Viz", layout="wide")
st.title("🔬 Clinical Trial QA with Gemini + Visualization")

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