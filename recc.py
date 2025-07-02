import streamlit as st

from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# 1. Define schema
class PromptInputState(TypedDict):
    input: dict
    output: str

# 2. Setup Groq LLM
llm = ChatGroq(
    temperature=0.3,
    

    groq_api_key = st.secrets["GROQ_API_KEY"],
    model_name="llama3-70b-8192"
)

# 3. Define prompt template
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful prompt engineering assistant.

Create a structured and effective prompt based on the following details:

Task: {task}  
Tone: {tone}  
Context: {context}

Respond in this format:

### Task  
{task}

### Tone  
{tone}

### Context  
{context}

---

**Prompt:**  
[Generated prompt for an LLM to use]
""")

# 4. Create LangGraph node
def generate_prompt(state: PromptInputState):
    formatted_prompt = prompt_template.format(**state["input"])
    response = llm.invoke(formatted_prompt)
    return {"input": state["input"], "output": response.content}

# 5. Define graph
builder = StateGraph(PromptInputState)
builder.add_node("generate", generate_prompt)
builder.set_entry_point("generate")
builder.set_finish_point("generate")
graph = builder.compile()

# 6. Streamlit UI
st.set_page_config(page_title="PromptCrafter", layout="centered")
st.title("🧠 PromptCrafter – Smart Prompt Generator")

with st.form("prompt_form"):
    task = st.text_input("📝 Task (e.g., write an email, generate an image prompt)")
    tone = st.text_input("🎯 Tone (e.g., formal, creative, humorous)")
    context = st.text_area("📘 Context (e.g., platform, length, specific instructions)")
    submitted = st.form_submit_button("Generate Prompt")

if submitted:
    state = {
        "input": {
            "task": task,
            "tone": tone,
            "context": context
        },
        "output": ""
    }
    result = graph.invoke(state)
    st.markdown("---")
    st.subheader("🧾 Structured Prompt")
    st.markdown(result["output"])
