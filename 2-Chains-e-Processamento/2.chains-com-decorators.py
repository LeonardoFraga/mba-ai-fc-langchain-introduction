from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnable import chain
from dotenv import load_dotenv
load_dotenv()

@chain
def square(input_dict: dict) -> int:
    x = input_dict["x"]
    return {"square_result": x * x}

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi I am {name}, tell me a joke with my name"
)

question_template2 = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me about the number {square_result}"
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain = question_template | model
chain2 = square | question_template2 | model

result = chain2.invoke({"x": 16})

# result = chain.invoke({"name": "Leonardo"})

print(result.content)