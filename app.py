# app.py
import streamlit as st
from rag_core.chain import get_response
from dotenv import load_dotenv

# Load env vars (Groq, Pinecone keys)
load_dotenv()

st.set_page_config(page_title="LoL Nexus", page_icon="⚔️")

st.title("⚔️ LoL Nexus: Real-Time Patch Analyst")
st.caption("Powered by Groq (Llama-3) & Pinecone Serverless")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "latency" in message:
            st.caption(f"⚡ Latency: {message['latency']}ms")

# React to user input
if prompt := st.chat_input("Ask about Patch 14.1 changes..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing patch notes..."):
            result = get_response(prompt)
            answer = result["answer"]
            latency = result["latency_ms"]
            
            st.markdown(answer)
            st.caption(f"⚡ Inference Latency: {latency}ms")
            
    # Add to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "latency": latency
    })