

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


# Prompt from the original paper https://arxiv.org/pdf/2305.14283
# Query Rewriting for Retrieval-Augmented Large Language Models
class QueryDecomposition(BaseModel):
    """
    Decompose the user query into smaller parts to think step by step to answer this question
    Act as a simple planning agent
    """

    questions: List[str] = Field(
        description="""
        Think step by step to answer this question, and provide one or several search engine questions in English for knowledge that you need. 
        Suppose that the user is looking for information about climate change, energy, biodiversity, nature, and everything we can find the IPCC reports and scientific literature
        - If it's already a standalone question, you don't need to provide more questions, just reformulate it if relevant as a better question for a search engine 
        - If you need to decompose the question, output a list of maximum 3 questions
    """
    )


class Location(BaseModel):
    country:str = Field(...,description="The country if directly mentioned or inferred from the location (cities, regions, adresses), ex: France, USA, ...")
    location:str = Field(...,description="The specific place if mentioned (cities, regions, addresses), ex: Marseille, New York, Wisconsin, ...")

class QueryAnalysis(BaseModel):
    """
    Analyzing the user query to extract topics, sources and date
    Also do query expansion to get alternative search queries
    Also provide simple keywords to feed a search engine
    """

    # keywords: List[str] = Field(
    #     description="""
    #     Extract the keywords from the user query to feed a search engine as a list
    #     Maximum 3 keywords

    #     Examples:
    #     - "What is the impact of deep sea mining ?" -> deep sea mining
    #     - "How will El Nino be impacted by climate change" -> el nino;climate change
    #     - "Is climate change a hoax" -> climate change;hoax
    #     """
    # )

    # alternative_queries: List[str] = Field(
    #     description="""
    #     Generate alternative search questions from the user query to feed a search engine
    #     """
    # )

    # step_back_question: str = Field(
    #     description="""
    #     You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.
    #     This questions should help you get more context and information about the user query
    #     """
    # )

    sources: List[Literal["IPCC", "IPBES", "IPOS"]] = Field( #,"OpenAlex"]] = Field(
        ...,
        description="""
            Given a user question choose which documents would be most relevant for answering their question,
            - IPCC is for questions about climate change, energy, impacts, and everything we can find the IPCC reports
            - IPBES is for questions about biodiversity and nature
            - IPOS is for questions about the ocean and deep sea mining
        """,
            # - OpenAlex is for any other questions that are not in the previous categories but could be found in the scientific litterature 
    )
    # topics: List[Literal[
    #     "Climate change",
    #     "Biodiversity",
    #     "Energy",
    #     "Decarbonization",
    #     "Climate science",
    #     "Nature",
    #     "Climate policy and justice",
    #     "Oceans",
    #     "Deep sea mining",
    #     "ESG and regulations",
    #     "CSRD",
    # ]] = Field(
    #     ...,
    #     description = """
    #         Choose the topics that are most relevant to the user query, ex: Climate change, Energy, Biodiversity, ...
    #     """,
    # )
    # date: str = Field(description="The date or period mentioned, ex: 2050, between 2020 and 2050")
    # location:Location


def make_query_decomposition_chain(llm):

    openai_functions = [convert_to_openai_function(QueryDecomposition)]
    llm_with_functions = llm.bind(functions = openai_functions,function_call={"name":"QueryDecomposition"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant, you will analyze, translate and reformulate the user input message using the function provided"),
        ("user", "input: {input}")
    ])

    chain = prompt | llm_with_functions | JsonOutputFunctionsParser()
    return chain


def make_query_rewriter_chain(llm):

    openai_functions = [convert_to_openai_function(QueryAnalysis)]
    llm_with_functions = llm.bind(functions = openai_functions,function_call={"name":"QueryAnalysis"})



    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant, you will analyze, translate and reformulate the user input message using the function provided"),
        ("user", "input: {input}")
    ])


    chain = prompt | llm_with_functions | JsonOutputFunctionsParser()
    return chain


def make_query_transform_node(llm):

    decomposition_chain = make_query_decomposition_chain(llm)
    rewriter_chain = make_query_rewriter_chain(llm)
        
    def transform_query(state):
        
        new_state = {}
            
        # Decomposition
        decomposition_output = decomposition_chain.invoke({"input":state["query"]})
        new_state.update(decomposition_output)
        
        # Query Analysis
        questions = []
        for question in new_state["questions"]:
            question_state = {"question":question}
            analysis_output = rewriter_chain.invoke({"input":question})
            
            # The case when the llm does not return any sources
            if not analysis_output["sources"] or not all(source in ["IPCC", "IPBS", "IPOS"] for source in analysis_output["sources"]):
                analysis_output["sources"] = ["IPCC", "IPBES", "IPOS"]

            question_state.update(analysis_output)
            questions.append(question_state)
        new_state["questions"] = questions
        
        return new_state
    
    return transform_query