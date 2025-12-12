from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi I am {name}, tell me a joke with my name"
)


model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain = question_template | model

result = chain.invoke({"name": "Leonardo"})
print(result.content)