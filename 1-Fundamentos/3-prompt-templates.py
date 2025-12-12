from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Hi I am {name}, tell me a joke with my name"
)

text = template.format(name="Leonardo")

print(text)