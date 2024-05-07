import streamlit as st
from functions.sidebar import sidebar
from functions.web_chain import vectorize

sidebar()

st.title("Upload Data")

with st.form(key='my_form'):

    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True,label_visibility="hidden")

    if "pdfs" not in st.session_state:
        st.session_state.pdfs = []
 
    url = st.text_input("Enter a website link")

    if "websites" not in st.session_state:
        st.session_state.websites = []

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if url:
        st.session_state.websites.append(url)
        vectorstore = vectorize(url)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        answer = retriever.invoke("What are promises?")
        for value in answer:
            st.write(value.page_content)
        
    if uploaded_files:
        st.session_state.pdfs.append(uploaded_files)