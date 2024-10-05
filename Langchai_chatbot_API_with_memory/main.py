from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI

from langchain_groq import ChatGroq
import os
os.environ["GROQ_API_KEY"]= os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-8b-8192",temperature =0)

# Initialize FastAPI app
app = FastAPI()

# Define the input model for request body
class ChatInput(BaseModel):
    text: str

# Initialize Langchain components (prompt, memory, chain)
prompt = ChatPromptTemplate(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

legacy_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

# Endpoint to process text input and return the LLM response
@app.post("/process/")
async def process_input(chat_input: ChatInput):
    try:
        # Invoke the chain with user input
        result = legacy_chain.invoke({"text": chat_input.text})
        return {"response": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to retrieve chat history
@app.get("/history/")
async def get_chat_history():
    try:
        # Fetch chat history from memory
        history = memory.chat_memory.messages
        formatted_history = [{"role": msg.type, "content": msg.content} for msg in history]
        return {"chat_history": formatted_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn app_name:app --reload
