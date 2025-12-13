from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Depending on the size of the history, we may want to trim the oldest messages.

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant that answers with a short joke when possible."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", temperature=0.9)

def prepare_input(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])

    trimmed = trim_messages(
        raw_history,
        token_counter=len,
        max_tokens=2,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    return {"input": payload.get("input", ""), "history": trimmed}


prepare = RunnableLambda(prepare_input)
chain = prepare | prompt | llm


session_store: dict[str, InMemoryChatMessageHistory] = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history",
)

config = {"configurable": {"session_id": "demo-session"}}

# Interactions

resp1 = conversational_chain.invoke({"input": "My name is Leonardo. Reply only with 'OK' and do not mention my name."}, config=config)
print("Assistant:", resp1.content)

resp2 = conversational_chain.invoke({"input": "Tell me a one-sentence fun fact. Do not mention my name."}, config=config)
print("Assistant:", resp2.content)

resp3 = conversational_chain.invoke({"input": "What is my name?"}, config=config)
print("Assistant:", resp3.content)