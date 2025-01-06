from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in environment variables")

# Embed the documents
# Configuration variables
embedding_model = "text-embedding-3-small"
doc_source_path = 'knowledgebase/'
chunk_size = 1800
chunk_overlap = 300

# Load txt documents
text_loader = DirectoryLoader(
    doc_source_path,
    glob="**/*.txt", 
    loader_cls=TextLoader, 
)
pdf_loader = DirectoryLoader(
    doc_source_path,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
)

text_documents = text_loader.load()
pdf_documents = pdf_loader.load()
documents = text_documents + pdf_documents

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
texts = text_splitter.split_documents(documents)

# Initialize embedding model
embeddings = OpenAIEmbeddings(model=embedding_model)

# Set up vectorstore
vectorstore = Qdrant.from_documents(
    texts, embeddings,
    location=":memory:",
    collection_name="PMarca", 
)

# Initiate document retriever
retriever = vectorstore.as_retriever()

# Setting up caching system locally
set_llm_cache(SQLiteCache(database_path="cache.db"))

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=800, top_p=1)

# Define the RAG chain
prompt = PromptTemplate.from_template(
    """Given {context}, answer the question `{question}` as bullet points. Most of your response should come from the {context}.
    Be creative, concise and be as practical as possible"""
)
rag_chain = (
    RunnablePassthrough.assign(context=lambda inputs: [doc.page_content for doc in retriever.invoke(inputs["question"])])
    | prompt
    | llm
    | StrOutputParser()
)

# Define Pydantic model for requests
class QuestionRequest(BaseModel):
    question: str

# API endpoint to handle questions
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        response = rag_chain.invoke({"question": request.question})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#run this on the terminal uvicorn main:app --reload