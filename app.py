import streamlit as st
import pandas as pd
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import plotly.express as px

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Data Analyst Bot", layout="wide")
st.title("📊 AI Data Analyst Bot")

# ---------------- SESSION STATE ---------------- #
if "plot_counter" not in st.session_state:
    st.session_state.plot_counter = 0

# ---------------- SAFE PLOT FUNCTION ---------------- #
def safe_plot(fig):
    st.session_state.plot_counter += 1
    st.plotly_chart(fig, key=f"plot_{st.session_state.plot_counter}")

# ---------------- LOAD API ---------------- #
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ---------------- CLEAN CODE ---------------- #
def clean_code(code):
    code = re.sub(r"```python", "", code)
    code = code.replace("```", "")
    code = re.sub(r"import .*", "", code)  # remove unsafe imports
    return code.strip()

# ---------------- FILE UPLOAD ---------------- #
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data")

    if question:

        # reset plot counter for each question
        st.session_state.plot_counter = 0

        # ---------------- PROMPT ---------------- #
        prompt = f"""
You are a senior data analyst.

Dataset columns: {list(df.columns)}

Write Python pandas code to answer the question.

Rules:
- Dataframe name is df
- Store final answer in variable 'result'
- Use plotly.express as px for charts
- DO NOT import anything
- If plotting:
    - create fig
- No print()
- No explanation
- No markdown

Question: {question}
"""

        response = llm.invoke(prompt)

        code = clean_code(response.content)

        st.subheader("🧠 Generated Code")
        st.code(code)

        # ---------------- EXECUTION ---------------- #
        try:
            local_vars = {
                "df": df,
                "px": px,
                "plot": safe_plot
            }

            safe_globals = {
                "__builtins__": {
                    "len": len,
                    "range": range,
                    "min": min,
                    "max": max,
                    "sum": sum
                }
            }

            exec(code, safe_globals, local_vars)

            # ---------------- RESULT ---------------- #
            if "result" in local_vars:
                st.subheader("📊 Answer")
                st.write(local_vars["result"])
            else:
                st.warning("No result variable found.")

        except Exception as e:
            st.error(f"Execution Error: {e}")
            