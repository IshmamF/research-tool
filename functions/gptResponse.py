from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

def get_response(user_query, chat_history, context):
    template = """
    You are a helpful assistant. Answer the following questions considering the background information of the conversation:

    Chat History: {chat_history}

    Background Information: {context}

    User question: {user_question}
    """


    llm = ChatOpenAI(api_key=openai_key)
    try:
        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI(api_key=openai_key)
            
        chain = prompt | llm | StrOutputParser()
        
        value = chain.stream({
            "chat_history": chat_history,
            "context": context,
            "user_question": user_query,
        })
        if value:
            response = " ".join([part for part in value])
            print(response)
            return response
        else:
            print("Empty response from API")
            return "No response received from model."
    except Exception as e:
        print(e)  
        return f"Error in generating response: {str(e)}"
