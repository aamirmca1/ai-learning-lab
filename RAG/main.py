import os
import uuid
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")

# Define a tool ---------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    print(f"Fetching weather for city: {city}")
    return f"It's always sunny in {city}!"

@tool
def get_news(topic: str) -> str:
    """Get news for a given topic."""
    print(f"Fetching news for topic: {topic}")
    return f"Here are the latest news articles about {topic}."

@tool
def get_word_length(word: str) -> int:
    """Get the length of a given word."""
    print(f"Calculating length for word: {word}")
    return len(word)

@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [get_weather, get_news, get_word_length, get_current_time]

# 1. Initialize the Endpoint first
endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=api_key,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.01,
)

# 2. Wrap it for chat interactions
llm = ChatHuggingFace(llm=endpoint)

# 3. Pass the model instance to the agent with Memory
memory = MemorySaver()
agent = create_agent(
            model=llm, 
            tools=tools,
            system_prompt= (
                "You are a helpful assistant. Use the tools provided to answer the user's questions."
            ),
            checkpointer=memory
            )


def chat():
    thread_id = str(uuid.uuid4())
    config = {
        "thread_id": thread_id
    }

    print("\n Welcome to the RAG Agent! Type 'exit' to quit.\n")
    print(f"Your thread ID: {thread_id}\n")

    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
            
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config
        )
        last_msg = result["messages"][-1].content_blocks
        print(f"Final Answer: {last_msg}")

if __name__ == "__main__":
    chat()