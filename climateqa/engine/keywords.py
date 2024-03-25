
from typing import List
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

class KeywordsOutput(BaseModel):
    """Analyzing the user query to get keywords for a search engine"""
    
    keywords: list = Field(
        description="""
        Generate 1 or 2 relevant keywords from the user query to ask a search engine for scientific research papers.
        
        Example:
        - "What is the impact of deep sea mining ?" -> ["deep sea mining"]
        - "How will El Nino be impacted by climate change" -> ["el nino"]
        - "Is climate change a hoax" -> [Climate change","hoax"]
        """
    )


def make_keywords_chain(llm):

    functions = [convert_to_openai_function(KeywordsOutput)]
    llm_functions = llm.bind(functions = functions,function_call={"name":"KeywordsOutput"})

    chain = llm_functions | JsonOutputFunctionsParser()
    return chain