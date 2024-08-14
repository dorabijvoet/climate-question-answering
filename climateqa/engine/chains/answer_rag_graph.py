from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from climateqa.engine.chains.prompts import answer_prompt_graph_template

class RecommendedGraph(BaseModel):
    title: str = Field(description="Title of the graph")
    embedding: str = Field(description="Embedding link of the graph")

# class RecommendedGraphs(BaseModel):
#     recommended_content: List[RecommendedGraph] = Field(description="List of recommended graphs")

def make_rag_graph_chain(llm):
    parser = JsonOutputParser(pydantic_object=RecommendedGraph)
    prompt = PromptTemplate(
        template=answer_prompt_graph_template,
        input_variables=["query", "recommended_content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    return chain

def make_rag_graph_node(llm):
    chain = make_rag_graph_chain(llm)

    def answer_rag_graph(state):
        output = chain.invoke(state)
        return {"graph_returned": output}

    return answer_rag_graph