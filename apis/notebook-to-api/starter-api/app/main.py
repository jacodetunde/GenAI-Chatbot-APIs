from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.rag import create_rag_chain
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()


# Define Pydantic model for requests
class QuestionRequest(BaseModel):
    question: str


# API endpoint to handle questions
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        rag_chain = create_rag_chain()
        response = rag_chain.invoke({"question": request.question})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
