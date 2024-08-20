
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


class IntentCategorizer(BaseModel):
    """Analyzing the user message input"""
    
    environment: bool = Field(
        description="Return 'True' if the question relates to climate change, the environment, nature, etc. (Example: should I eat fish?). Return 'False' if the question is just chit chat or not related to the environment or climate change.",
    )


def make_chitchat_intent_categorization_chain(llm):

    openai_functions = [convert_to_openai_function(IntentCategorizer)]
    llm_with_functions = llm.bind(functions = openai_functions,function_call={"name":"IntentCategorizer"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant, you will analyze, translate and reformulate the user input message using the function provided"),
        ("user", "input: {input}")
    ])

    chain = prompt | llm_with_functions | JsonOutputFunctionsParser()
    return chain


def make_chitchat_intent_categorization_node(llm):

    categorization_chain = make_chitchat_intent_categorization_chain(llm)

    def categorize_message(state):
        output = categorization_chain.invoke({"input": state["user_input"]})
        print(f"\n\nChit chat output intent categorization: {output}\n")
        state["search_graphs_chitchat"] = output["environment"]
        print(f"\n\nChit chat output intent categorization: {state}\n")
        return state
    
    return categorize_message
