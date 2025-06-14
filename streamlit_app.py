import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import nltk
import os
import shutil
from nltk import word_tokenize, pos_tag
import re

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
If the answer is not found, say "I am not sure about this answer, please check the database."
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
Ensure the code includes clear x axis ,y axis labels which are Descriptive and Accurate and Relevant, an informative title which is accurate, and uses the best-fit chart type (bar, pie, line, etc.) Make the graph detailed. The graph should be very relevant to the question.

{context_str}

Python code only:"""
    return model.generate_content(prompt).text

# --- Fixed: Extract Python Code Block ---
def extract_code_block(text):
    match = re.search(r"```python\s+(.*?)```", text, re.DOTALL)
    if not match:
        match = re.search(r"```(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None

# --- Streamlit UI ---
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

# --- Session State ---
if "prev_question" not in st.session_state:
    st.session_state.prev_question = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

selected_question = st.selectbox("❓ Select a Question", questions)

if selected_question != st.session_state.prev_question:
    st.session_state.answer = ""
    st.session_state.prev_question = selected_question

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    matched_df = row_matcher(df, selected_question)

    st.subheader("🧠 Gemini Q/A Answer")
    if st.session_state.answer:
        st.markdown(st.session_state.answer)

    if st.button("Generate Answer"):
        answer_text = generate_answer_only(selected_question, matched_df)
        st.session_state.answer = answer_text
        st.markdown(answer_text)

    if st.session_state.answer.strip() != "":
        st.subheader("📊 Graph Based on Answer")
        if st.button("Generate Graph from Answer"):
            full_context = f"""Matched Rows:\n{matched_df.to_string(index=False)}\n\nQuestion:\n{selected_question}\n\nAnswer:\n{st.session_state.answer}"""
            plot_code_response = generate_plot_code_from_answer(full_context)

            st.subheader("📈 Interactive Graph")
            try:
                fig = go.Figure()
                numeric_cols = matched_df.select_dtypes(include='number').columns
                category_col = matched_df.columns[0]

                colors = ['green', 'blue', 'orange', 'red', 'purple']
                for i, metric in enumerate(numeric_cols):
                    fig.add_trace(go.Bar(
                        y=matched_df[category_col],
                        x=matched_df[metric],
                        name=metric,
                        orientation='h',
                        marker_color=colors[i % len(colors)],
                        hovertemplate=f"%{{x}}%<extra>{metric}</extra>"
                    ))

                fig.update_layout(
                    title="Interactive Clinical Trial Metrics",
                    barmode='group',
                    xaxis_title="Percentage",
                    yaxis_title=category_col,
                    height=500,
                    legend_title="Metric",
                    margin=dict(l=100, r=20, t=50, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error generating interactive graph: {e}")
else:
    st.info("Upload an Excel file to begin.")
