from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from operator import itemgetter

from climateqa.engine.chains.prompts import answer_prompt_graph_template

# class RecommendedGraph(BaseModel):
#     title: str = Field(description="Title of the graph")
#     embedding: str = Field(description="Embedding link of the graph")


# class RecommendedGraphs(BaseModel):
#     recommended_content: List[RecommendedGraph] = Field(description="List of recommended graphs")

# def make_rag_graph_chain(llm):
#     parser = JsonOutputParser(pydantic_object=RecommendedGraph)
#     prompt = PromptTemplate(
#         template=answer_prompt_graph_template,
#         input_variables=["query", "recommended_content"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )

#     chain = prompt | llm | parser
#     return chain


def _format_graphs(recommended_content):

    graphs = []
    for x in recommended_content:
        embedding = x.metadata["returned_content"]
        
        # Check if the embedding has already been seen
        graphs.append({
            "title": x.page_content,
            "embedding": embedding,
            "metadata": {
                "source": x.metadata["source"],
                "category": x.metadata["category"]
            }
        })

    formatted_strings = []
    for item in graphs:
        title = item.get('title', 'N/A')
        embedding = item.get('embedding', 'N/A')
        source = item['metadata'].get('source', 'N/A')
        category = item['metadata'].get('category', 'N/A')
        
        formatted_string = f"Title: {title}\nEmbedding: {embedding}\nSource: {source}\nCategory: {category}\n"
        formatted_strings.append(formatted_string)
    
    return "\n".join(formatted_strings)


class Graph(BaseModel):
    embedding: str = Field(description="List of the relevant graphs' embedding code.")
    category: str = Field(description="Category of the graph")
    source: str = Field(description="Source of the graph")


class RecommendedGraphs(BaseModel):
    graphs: List[Graph] = Field(description="List of dictionaries of the relevant graphs.")

def make_rag_graph_chain(llm):
    input_documents = {
        "query" : itemgetter("query"),
        "recommended_content" : lambda x : _format_graphs(x["recommended_content"])
    }

    parser = JsonOutputParser(pydantic_object=RecommendedGraphs)

    prompt = PromptTemplate(
        template=answer_prompt_graph_template,
        input_variables=["query", "recommended_content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = input_documents | prompt | llm | parser
    return chain

def make_rag_graph_node(llm):
    chain = make_rag_graph_chain(llm)

    def answer_rag_graph(state):
        output = chain.invoke(state)
        return {"graphs_returned": output}

    return answer_rag_graph