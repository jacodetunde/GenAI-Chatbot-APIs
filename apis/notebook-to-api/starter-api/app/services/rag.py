from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyMuPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from app.config.settings import settings

# Set up local caching
set_llm_cache(SQLiteCache(database_path="cache.db"))


def initialize_vectorstore():
    # Load documents
    text_loader = DirectoryLoader(
        settings.DOC_SOURCE_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    pdf_loader = DirectoryLoader(
        settings.DOC_SOURCE_PATH,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
    )
    text_documents = text_loader.load()
    pdf_documents = pdf_loader.load()
    documents = text_documents + pdf_documents

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    texts = text_splitter.split_documents(documents)

    # Initialize embedding model and vectorstore
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vectorstore = Qdrant.from_documents(
        texts,
        embeddings,
        location=":memory:",
        collection_name="PMarca",
    )
    return vectorstore


def create_rag_chain():
    retriever = initialize_vectorstore().as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=800, top_p=1)

    prompt = PromptTemplate.from_template(
        """Given {context}, answer the question `{question}` as bullet points. Most of your response should come from the {context}.
        Be creative, concise and be as practical as possible"""
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda inputs: [
                doc.page_content for doc in retriever.invoke(inputs["question"])
            ]
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
