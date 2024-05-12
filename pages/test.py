import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from functions.gptResponse import get_response
from functions.sidebar import sidebar
from functions.web_chain import vectorize, loadUrlData, get_pdf_text, recursiveURL
import asyncio


async def main():
    sidebar()
    st.title("Upload Data")

    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    st.warning("If you plan to add more files, after processing initial files, make sure the uploaded files you already processed are removed")
    url = st.text_input("Enter a website link")
    if url:
        st.write(recursiveURL(url))
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = False
    if st.button('Process URL and Files'):
        st.session_state.button_pressed = True
        with st.spinner("Vectorizing Data, wait times vary depending on size..."):
            if url:
                try:
                    if "retriever" not in st.session_state:
                        st.session_state.retriever = vectorize(loadUrlData(url), "document")
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")
            if uploaded_files:
                try:
                    texts = get_pdf_text(uploaded_files)
                    if texts:
                        if "retriever" not in st.session_state:
                            st.session_state.retriever = vectorize(texts, "text")
                        else:
                            st.session_state.retriever.add_texts(texts)
                    else:
                        st.error("PDF has no meta data text")
                except Exception as e:
                    st.error(f"Failed to load PDF: {e}")
            st.success("Data is ready to be queried!")
    
    if st.session_state.button_pressed:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

        st.title("RAG CHAT")
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

        user_query = st.chat_input("Type your message here...", key="chat_input")
        if user_query:
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            with st.chat_message("Human"):
                st.write(user_query)

            if 'retriever' in st.session_state:
                try:
                    ragAnswer = await st.session_state.retriever.amax_marginal_relevance_search(user_query, k=4, fetch_k=10)
                    context = []
                    for i, doc in enumerate(ragAnswer):
                        print(f"{i}: {doc.page_content}")
                        context.append(doc.page_content)
                    with st.spinner("Generating Response"):
                        response = get_response(user_query, st.session_state.chat_history, context)
                    if response:
                        st.session_state.chat_history.append(AIMessage(content=response))
                        with st.chat_message("AI"):
                            st.write(response)
                    else:
                        st.write("No response received.") 
                except Exception as e:
                    st.error(f"Error during retrieval or response generation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
