
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


class Translation(BaseModel):
    """Analyzing the user message input"""
    
    translation: str = Field(
        description="Translate the message input to English",
    )


def make_translation_chain(llm):

    openai_functions = [convert_to_openai_function(Translation)]
    llm_with_functions = llm.bind(functions = openai_functions,function_call={"name":"Translation"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant, you will translate the user input message to English using the function provided"),
        ("user", "input: {input}")
    ])

    chain = prompt | llm_with_functions | JsonOutputFunctionsParser()
    return chain


def make_translation_node(llm):

    translation_chain = make_translation_chain(llm)

    def translate_query(state):
        user_input = state["user_input"]
        translation = translation_chain.invoke({"input":user_input})
        return {"query":translation["translation"]}

    return translate_query
