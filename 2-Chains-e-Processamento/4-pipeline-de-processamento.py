from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Define a prompt template for translation
# This template takes an input variable 'initial_text' and generates a prompt
# to translate the given text to English.
template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n ```{initial_text}````"
)

# Define a prompt template for summarization
# This template takes an input variable 'text' and generates a prompt
# to summarize the given text in 4 words.
template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words:\n ```{text}```"
)

llm_en = ChatOpenAI(model="gpt-5-mini", temperature=0)

# Create a translation chain
# This chain takes the translation template, processes it with the LLM,
# and parses the output as a string.
translate = template_translate | llm_en | StrOutputParser()

# Create a processing pipeline
# This pipeline first translates the input text using the translation chain,
# then summarizes the translated text using the summarization template and the LLM,
# and finally parses the output as a string.
pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

# Invoke the pipeline with an initial text in Portuguese
# The pipeline will translate the text to English and then summarize it in 4 words.
result = pipeline.invoke({"initial_text": "LangChain é um framework para desenvolvimento de aplicações de IA"})

print(result)