import sys
import os
from contextlib import contextmanager

from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod

from typing_extensions import TypedDict
from typing import List

from IPython.display import display, HTML, Image

from climateqa.engine.chains.answer_chitchat import make_chitchat_node
from climateqa.engine.chains.answer_ai_impact import make_ai_impact_node
from climateqa.engine.chains.query_transformation_with_IEA import make_query_transform_node
from climateqa.engine.chains.translation import make_translation_node
from climateqa.engine.chains.intent_categorization import make_intent_categorization_node
from climateqa.engine.chains.with_iea_retriever import make_cqa_retriever_node, make_iea_retriever_node
from climateqa.engine.chains.answer_rag import make_rag_node
from climateqa.engine.chains.answer_rag_with_iea import make_rag_node_recommended_content

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    user_input : str
    language : str
    intent : str
    query : str
    questions : List[dict]
    answer : str
    audience : str
    sources_input : List[str]
    documents : List[Document]
    recommended_content : List[Document]
    recommended_content_answer : str

def set_defaults(state):
    if not state["audience"] or state["audience"] is None:
        state.update({"audience": "experts"})   

    if not state["sources_input"] or state["sources_input"] is None:
        state.update({"sources_input": ["auto"]})  
                                 
    return state

def search(state):
    return {}

def route_intent(state):
    intent = state["intent"]
    if intent in ["chitchat","esg"]:
        return "answer_chitchat"
    elif intent == "ai":
        return "answer_ai_impact"
    else:
        # Search route
        return "search"
    
def route_translation(state):
    if state["language"].lower() == "english":
        return "transform_query"
    else:
        return "translate_query"

# Le routing qui définit si on répond avec "answer_rag" ou "answer_rag_without_docs"
def route_based_on_relevant_docs(state,threshold_docs=0.2):
    docs = [x for x in state["documents"] if x.metadata["reranking_score"] > threshold_docs]
    if len(docs) > 0:
        return "answer_rag"
    else:
        return "answer_rag_no_docs"
    

def make_id_dict(values):
    return {k:k for k in values}

def format_answer(state):
    final_answer = f"{state['answer']} \n\n\n {state['recommended_content_answer']}"
    return {"answer": final_answer}

def make_graph_agent_with_recommended_content(llm,cqa_vectorstore,iea_vectorstore,reranker,threshold_docs = 0.2):
    
    workflow = StateGraph(GraphState)

    # Define the node functions
    categorize_intent = make_intent_categorization_node(llm)
    transform_query = make_query_transform_node(llm) # DORA
    translate_query = make_translation_node(llm)
    answer_chitchat = make_chitchat_node(llm)
    answer_ai_impact = make_ai_impact_node(llm)
    retrieve_documents_cqa = make_cqa_retriever_node(cqa_vectorstore,reranker)
    retrieve_documents_iea = make_iea_retriever_node(iea_vectorstore,reranker)
    answer_rag = make_rag_node(llm,with_docs=True)
    answer_rag_no_docs = make_rag_node(llm,with_docs=False)
    answer_rag_with_recommended_content = make_rag_node_recommended_content(llm, with_docs=True)

    # Define the nodes
    workflow.add_node("set_defaults", set_defaults)
    workflow.add_node("categorize_intent", categorize_intent)
    workflow.add_node("search", search)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("translate_query", translate_query)
    workflow.add_node("answer_chitchat", answer_chitchat)
    workflow.add_node("answer_ai_impact", answer_ai_impact)
    workflow.add_node("retrieve_documents_cqa", retrieve_documents_cqa)
    workflow.add_node("retrieve_documents_iea", retrieve_documents_iea)
    workflow.add_node("answer_rag",answer_rag)
    workflow.add_node("answer_rag_no_docs",answer_rag_no_docs)
    workflow.add_node("answer_rag_with_recommended_content", answer_rag_with_recommended_content)
    workflow.add_node("format_answer", format_answer)

    # Entry point
    workflow.set_entry_point("set_defaults")

    # CONDITIONAL EDGES
    workflow.add_conditional_edges(
        "categorize_intent",
        route_intent,
        make_id_dict(["answer_chitchat","answer_ai_impact","search"])
    )

    workflow.add_conditional_edges(
        "search",
        route_translation,
        make_id_dict(["translate_query","transform_query"])
    )

    workflow.add_conditional_edges(
        "retrieve_documents_iea",
        lambda x : route_based_on_relevant_docs(x,threshold_docs=threshold_docs),
        make_id_dict(["answer_rag","answer_rag_no_docs"])
    )

    # workflow.add_conditional_edges(
    #     "answer_rag",
    #     route_recommended_content,
    #     make_id_dict(["answer_rag_with_recommended_content", "answer_rag_without_recommended_content"])
    # )

    # Define the edges
    workflow.add_edge("set_defaults", "categorize_intent")
    workflow.add_edge("translate_query", "transform_query")
    workflow.add_edge("transform_query", "retrieve_documents_cqa")
    workflow.add_edge("retrieve_documents_cqa", "retrieve_documents_iea")
    workflow.add_edge("answer_rag", "answer_rag_with_recommended_content")
    workflow.add_edge("answer_rag_no_docs", "answer_rag_with_recommended_content")
    workflow.add_edge("answer_rag_with_recommended_content", "format_answer")
    workflow.add_edge("format_answer", END)
    workflow.add_edge("answer_chitchat", END)
    workflow.add_edge("answer_ai_impact", END)

    
    # Compile
    app = workflow.compile()
    return app