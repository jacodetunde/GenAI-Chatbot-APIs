import os
import jwt
import datetime
import boto3
from pydantic import BaseModel
from pymongo import MongoClient
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Union, Any, Text, Dict, List, AsyncGenerator, Annotated
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
openai_api_key: str = os.getenv("OPENAI_API_KEY", "my_api_key")
client_secret: str = os.getenv("CLIENT_SECRET", "my_client_secret")
mongo_connection_string: Optional[str] = os.getenv("MONGO_CONNECTION_STRING")

# bedrock setup
aws_region = os.getenv("AWS_REGION", "us-east-1")
prompt_id = os.getenv("BEDROCK_PROMPT_ID", "P4KU1GHFY0")
knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID", "DMEUZKFMFU")
prompt_version = os.getenv("BEDROCK_PROMPT_VERSION", "1")
knowledge_base_similarity_threshold = os.getenv(
    "KNOWLEDGE_BASE_SIMILARITY_THRESHOLD", "0.5"
)
# MongoDB Setup
mongo_client = MongoClient(mongo_connection_string)
genai_db = mongo_client["chatbot"]
conversation_collection = genai_db["tutorial-bot-conversations"]

openai_client = OpenAI(api_key=openai_api_key)
default_max_tokens = 4096
default_temperature = 0.7
top_p = 0.5
default_model = "gpt-4o"

string_padding = "<<<" + (" " * 1000) + ">>>"

bedrock_client = boto3.client("bedrock-agent", region_name=aws_region)
bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=aws_region)
num_knowledge_base_results = 3


def authenticate(auth_token: Any) -> Optional[Any]:
    bearer_token: str = auth_token.replace("Bearer ", "")
    output_payload: Dict[str, Any] = jwt.decode(
        bearer_token, client_secret, algorithms=["HS256"]
    )
    if "person_id" in output_payload:
        return str(output_payload["person_id"])

    return None


def get_prompt_template(client: Any, prompt_id: Text, prompt_version: Text) -> Any:
    try:
        response = client.get_prompt(
            promptIdentifier=prompt_id, promptVersion=prompt_version
        )
        template_text = response["variants"][0]["templateConfiguration"]["text"]["text"]
        logger.info(f"Successfully fetched prompt template: {template_text}")
        return template_text

    except Exception as e:
        logger.error(f"An error occurred while fetching the prompt: {e}")

        return """
        Provide answers to the user's question as bullet points. 
        Most of your response should come from the {context}. 
        Be creative, concise, and as practical as possible.
        """


def retrieve_knowledgebase_context(
    query: Text, knowledge_base_id: Text
) -> Optional[Text]:
    try:
        knowledge_base_response = bedrock_agent_client.retrieve(
            retrievalQuery={"text": query},
            knowledgeBaseId=knowledge_base_id,
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": num_knowledge_base_results,
                }
            },
        )

        retrieval_results = knowledge_base_response["retrievalResults"]
        contexts = []
        for result in retrieval_results:
            logger.info(result["score"])
            if result["score"] > float(knowledge_base_similarity_threshold):
                contexts.append(result["content"]["text"])

        if len(contexts) > 0:
            return "\n".join(contexts)

        return None

    except Exception as e:
        logger.error(
            f"An error occurred while performing the knowledge base retrieval: {e}"
        )
        return None


def load_conversation_history(person_id: str) -> List[Dict[str, Any]]:
    user_conversation = conversation_collection.find_one({"person_id": person_id})
    return user_conversation.get("messages", []) if user_conversation else []


def save_conversation_history(person_id: str, history: List[Dict[str, Any]]) -> None:
    conversation_collection.update_one(
        {"person_id": person_id},
        {"$set": {"messages": history}},
        upsert=True,
    )


def save_user_feedback(
    person_id: str, bot_message_text: str, user_feedback: str
) -> bool:
    found_message = False
    user_conversation = conversation_collection.find_one({"person_id": person_id})
    if not user_conversation or "messages" not in user_conversation:
        return False

    messages = user_conversation["messages"]

    # Find the bot message and update feedback
    for idx in reversed(range(len(messages))):
        message = messages[idx]
        if (
            message.get("role") == "assistant"
            and message.get("content", "").strip()[:100]
            == bot_message_text.strip()[:100]
        ):
            message["feedback"] = user_feedback
            found_message = True
            break

    if not found_message:  # Move this outside the loop
        return False

    save_conversation_history(person_id, messages)
    return True  # Return True if feedback was saved successfully


class UserRequest(BaseModel):  # type: ignore
    UserInput: Optional[str]
    maxTokens: int = default_max_tokens
    temperature: float = default_temperature
    model: str = default_model


class FeedBackRequest(BaseModel):  # type: ignore
    bot_message: str
    user_feedback: str


@app.post("/chat_process")  # type: ignore
def chat_process(
    user_request: UserRequest,
    Authorization: Union[str, None] = Header(None),
) -> Any:
    person_id = authenticate(Authorization)
    if not person_id:
        return {"error": "Unauthorized or invalid token"}

    message_list = [{"sender": "user", "text": user_request.UserInput}]
    return StreamingResponse(chat_completion(message_list, person_id))


async def chat_completion(
    message_list: List[Dict[str, Any]], person_id: str
) -> AsyncGenerator[str, None]:
    try:
        # Extract user input
        user_input = message_list[-1]["text"]

        # Retrieve knowledge base context
        context = retrieve_knowledgebase_context(
            query=user_input, knowledge_base_id=knowledge_base_id
        )
        if context:
            logger.info(f"Context used for chat completion: {context}")
        else:
            logger.warning("No context retrieved for chat completion.")

        # Format user input with context
        user_message_with_context = (
            f"Context: {context}\nUser Message: {user_input}" if context else user_input
        )

        # Load conversation history
        user_conversation_history = load_conversation_history(person_id)

        # Fetch and format the system prompt
        system_prompt = get_prompt_template(
            bedrock_client, prompt_id, prompt_version
        ).format(context=context)
        if not user_conversation_history:
            user_conversation_history = [{"role": "system", "content": system_prompt}]
        elif user_conversation_history[0].get("role") == "system":
            user_conversation_history[0]["content"] = system_prompt
            user_conversation_history[0]["timestamp"] = datetime.datetime.now()
        else:
            user_conversation_history.insert(
                0, {"role": "system", "content": system_prompt}
            )

        # Add user input to conversation history
        user_conversation_history.append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.datetime.now(),
            }
        )

        # Format conversation history for OpenAI API
        message_context_limit = 5  # Limit the context to the last 5 messages
        formatted_message = [
            {"role": m["role"], "content": m["content"]}
            for m in user_conversation_history
            if isinstance(m.get("content"), str)
            and m.get("role") in ["system", "user", "assistant"]
        ]
        message_list_formatted = formatted_message[-message_context_limit:]

        # Call OpenAI API for response generation
        response = openai_client.chat.completions.create(
            messages=message_list_formatted,
            model=default_model,
            temperature=default_temperature,
            max_tokens=default_max_tokens,
            top_p=top_p,
            stream=True,
        )
        response_text = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content + string_padding

        # Save assistant response to conversation history
        user_conversation_history.append(
            {
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.datetime.now(),
            }
        )

        # Save updated conversation history to MongoDB
        save_conversation_history(person_id, user_conversation_history)

    except Exception as e:
        logger.error(f"Error in chat_completion: {e}")
        yield "Error occurred while processing the request."


@app.post("/save_feedback")  # type: ignore
def save_feedback(
    feedback_request: FeedBackRequest,
    Authorization: Annotated[Union[str, None], Header()] = None,
) -> StreamingResponse:
    logger.info(f"Received feedback: {feedback_request}")
    person_id = authenticate(Authorization)
    if person_id:
        logger.info(f"Authenticated user: {person_id}")
        success = save_user_feedback(
            person_id=person_id,
            bot_message_text=feedback_request.bot_message,
            user_feedback=feedback_request.user_feedback,
        )
        if success:
            return StreamingResponse(content="Success", status_code=200)
        else:
            return StreamingResponse(content="Failed to save feedback", status_code=404)

    else:
        return StreamingResponse(content="Unauthorized", status_code=401)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
