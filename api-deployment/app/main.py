import os
import jwt
from pydantic import BaseModel
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from config.settings import settings
from dotenv import load_dotenv
from typing import Optional, Union, Any, Dict, List, AsyncGenerator
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "my_api_key")
client_secret = os.getenv("CLIENT_SECRET", "my_client_secret")

openai_client = OpenAI(api_key=openai_api_key)
default_max_tokens = 4096
default_temperature = 0.7
message_context_limit = 5
top_p = 0.5
default_model = "gpt-4o"

string_padding = "<<<" + (" " * 1000) + ">>>"

# Global variable for vectorstore
vectorstore = None


def authenticate(auth_token: Any) -> Optional[Any]:
    bearer_token: str = auth_token.replace("Bearer ", "")
    output_payload: Dict[str, Any] = jwt.decode(bearer_token, client_secret, algorithms=["HS256"])
    if "person_id" in output_payload:
        return str(output_payload["person_id"])


def get_vectorstore():
    global vectorstore
    if vectorstore is not None:
        return vectorstore  # Reuse the existing vectorstore

    try:
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
            texts, embeddings,
            location=":memory:",
            collection_name="PMarca",
        )
        print("Vector store initialized successfully.")
        return vectorstore
    except Exception as e:
        print(f"Error initializing vectorstore: {e}")
        raise RuntimeError("Failed to initialize vectorstore")


def get_prompt():
    return (
        """Provide answers to the user's question as bullet points. Most of your response should come from the {context}.
        Be creative, concise, and as practical as possible."""
    )


class UserRequest(BaseModel):
    UserInput: Optional[str]
    maxTokens: int = default_max_tokens
    temperature: float = default_temperature
    model: str = default_model
    document: Optional[str] = None


@app.post("/chat_process")
def chat_process(
    user_request: UserRequest,
    Authorization: Union[Any, None] = Header(None)
) -> Any:
    person_id = authenticate(Authorization)
    if person_id:
        return {"error": "Unauthorized or invalid token"}
    message_list = [{"sender": "user", "text": user_request.UserInput}]
    return StreamingResponse(chat_completion(message_list))


async def chat_completion(message_list: List[Any]) -> AsyncGenerator[str, None]:
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()

    try:
        # Extract user input and retrieve context
        user_input = message_list[-1]["text"]
        context_documents = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in context_documents])

        # Format the system prompt with context
        system_prompt = get_prompt().format(context=context)
        message_list_formatted = [{"role": "system", "content": system_prompt}] + [
            {"role": m["sender"], "content": m["text"]} for m in message_list
        ]

        # Call OpenAI API
        response_text = ""
        response = openai_client.chat.completions.create(
            messages=message_list_formatted,
            model=default_model,
            temperature=default_temperature,
            max_tokens=default_max_tokens,
            top_p=top_p,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content + string_padding
    except Exception as e:
        print(f"Error in chat_completion: {e}")
        yield "Error occurred while processing the request."


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
