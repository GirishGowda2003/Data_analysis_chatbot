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

# ---------------- LOAD API KEY ---------------- #
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input("Groq API Key",
        value=groq_api_key,
        type="password",
        placeholder="gsk_..."
    )
    st.markdown("[Get your free Groq API key](https://console.groq.com/keys)", unsafe_allow_html=True)

groq_api_key = api_key_input.strip()

# ---------------- CLEAN CODE ---------------- #
def clean_code(code):
    code = re.sub(r"```python", "", code)
    code = code.replace("```", "")
    code = re.sub(r"import .*", "", code)  # remove unsafe imports
    return code.strip()

# ---------------- FILE UPLOAD ---------------- #
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
    st.stop()

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data")

    if question:

        # reset plot counter for each question
        st.session_state.plot_counter = 0

        # ---------------- INIT LLM ---------------- #
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile"
            )
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            st.stop()

        # ---------------- PROMPT ---------------- #
        prompt = f"""
You are a senior data analyst.

Dataset columns: {list(df.columns)}
Dataset dtypes: {df.dtypes.to_dict()}

Write Python pandas code to answer the question.

Rules:
- Dataframe name is df
- Store final answer in variable 'result'
- Use plotly.express as px for charts
- DO NOT import anything
- If plotting:
    - create fig using px
    - call plot(fig) to display it
    - set result = fig
- No print()
- No explanation
- No markdown

Question: {question}
"""

        with st.spinner("🤖 Thinking..."):
            try:
                response = llm.invoke(prompt)
            except Exception as e:
                st.error(f"API Error: {e}")
                st.stop()

        code = clean_code(response.content)

        st.subheader("🧠 Generated Code")
        st.code(code)

        # ---------------- EXECUTION ---------------- #
        try:
            import builtins

            local_vars = {
                "df": df,
                "pd": pd,
                "px": px,
                "plot": safe_plot
            }

            safe_globals = {
                "__builtins__": builtins.__dict__
            }

            exec(code, safe_globals, local_vars)

            # ---------------- RESULT ---------------- #
            if "result" in local_vars:
                result = local_vars["result"]
                st.subheader("📊 Answer")
                # Don't re-render plotly figs (already shown via plot())
                if not hasattr(result, "to_plotly_json"):
                    st.write(result)
            else:
                st.warning("No result variable found.")

        except Exception as e:
            st.error(f"Execution Error: {e}")
