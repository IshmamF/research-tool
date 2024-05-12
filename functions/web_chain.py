from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup as Soup

def recursiveURL(url):
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
    )
    return loader.load()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def loadUrlData(url):
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {'verify':False}
    html = loader.load()
    return html

def loadUrlAllPages(link):
    crawl_params = {
        'crawlerOptions': {
            'excludes': ['blog/*'],
            'includes': [], # leave empty for all pages
            'limit': 1000,
        }
    }
    loader = FireCrawlLoader(
    api_key="fc-4dfc3630c705434893adfb46d1fe3c8a",
    url=link,
    mode="crawl",params=crawl_params)
    print(loader.load())
    return loader.load()




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






    