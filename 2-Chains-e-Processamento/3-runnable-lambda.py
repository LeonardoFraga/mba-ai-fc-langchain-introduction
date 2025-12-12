from langchain_core.runnable import RunnableLambda

def parse_number(text:str) -> int:
    return int(text.strip())

parse_runnable = RunnableLambda(parse_number)

number = parse_runnable.invoke("10")