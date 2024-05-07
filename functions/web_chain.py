from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def loadUrlData(url):
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {'verify':False}
    html = loader.load()
    return html

def splitDoc(data):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(data)

def splitText(data):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
    return text_splitter.split_text(data)

def vectorize(data, type):
    if type == "document":
        docs = splitDoc(data)
        return Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    elif type == "text":
        texts = splitText(data)
        return Chroma.from_texts(texts=texts, embedding=OpenAIEmbeddings())






    