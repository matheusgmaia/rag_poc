import asyncio
from typing import List

from fastapi import (
    FastAPI,
    HTTPException,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from pydantic import BaseModel

from .langchain_setup import (
    create_vector_store,
    rag_chain,
)

app = FastAPI(
    title="RAG Chatbot API", description="An API for a Retrieval-Augmented Generation chatbot with citations."
)

# --- Pydantic Models for Request and Response ---


class ChatHistoryMessage(BaseModel):
    role: str  # "human" or "ai"
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: List[ChatHistoryMessage] = []


class Citation(BaseModel):
    text: str
    source: int


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]


class UpdateResponse(BaseModel):
    status: str
    message: str


# --- API Endpoints ---


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a question and chat history, returns a RAG-generated answer with citations.
    """
    try:
        # Convert Pydantic models to LangChain message objects
        lc_chat_history = []
        for msg in request.chat_history:
            if msg.role.lower() == "human":
                lc_chat_history.append(HumanMessage(content=msg.content))
            elif msg.role.lower() == "ai":
                lc_chat_history.append(AIMessage(content=msg.content))

        # Prepare the input for the RAG chain
        chain_input = {"input": request.question, "chat_history": lc_chat_history}

        # Invoke the RAG chain asynchronously to avoid blocking the server
        result = await asyncio.to_thread(rag_chain.invoke, chain_input)

        return ChatResponse(answer=result.get("answer", ""), citations=result.get("citations", []))
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the chat request.") from e


@app.post("/update-embeddings", response_model=UpdateResponse)
async def update_embeddings_endpoint():
    """
    Triggers an update of the vector store embeddings from the files in the data folder.
    Note: This is a blocking operation and will make the server unresponsive until it completes.
    """
    try:
        await asyncio.to_thread(create_vector_store)
        return UpdateResponse(
            status="success",
            message="Vector store update process initiated.",
        )
    except Exception as e:
        print(f"An error occurred during update: {e}")
        raise HTTPException(status_code=500, detail="Failed to update embeddings.") from e
