import streamlit as st

def sidebar():

    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.page_link("pages/chat_rag.py", label="RAG CHAT")
    st.sidebar.page_link("pages/test.py", label="TESTING")

