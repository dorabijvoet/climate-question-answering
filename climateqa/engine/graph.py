import sys
import os
from contextlib import contextmanager

from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod

from typing_extensions import TypedDict
from typing import List, Dict

from IPython.display import display, HTML, Image

from climateqa.engine.chains.answer_chitchat import make_chitchat_node
from climateqa.engine.chains.answer_ai_impact import make_ai_impact_node
from climateqa.engine.chains.query_transformation import make_query_transform_node
from climateqa.engine.chains.translation import make_translation_node
from climateqa.engine.chains.intent_categorization import make_intent_categorization_node
from climateqa.engine.chains.retriever import make_retriever_node
from climateqa.engine.chains.answer_rag import make_rag_node
from climateqa.engine.chains.answer_rag_graph import make_rag_graph_node
from climateqa.engine.chains.set_defaults import set_defaults
from climateqa.engine.chains.graph_retriever import make_graph_retriever_node


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    user_input : str
    language : str
    intent : str
    query: str
    questions : List[dict]
    answer: str
    audience: str
    sources_input: List[str]
    documents: List[Document]
    recommended_content : List[Document]
    graph_returned: Dict[str,str]

def search(state):
    return {}

def route_intent(state):
    intent = state["intent"]
    if intent in ["chitchat","esg"]:
        return "answer_chitchat"
    elif intent == "ai_impact":
        return "answer_ai_impact"
    else:
        # Search route
        return "search"
    
def route_translation(state):
    if state["language"].lower() == "english":
        return "transform_query"
    else:
        return "translate_query"
    
def route_based_on_relevant_docs(state,threshold_docs=0.2):
    docs = [x for x in state["documents"] if x.metadata["reranking_score"] > threshold_docs]
    if len(docs) > 0:
        return "answer_rag"
    else:
        return "answer_rag_no_docs"
    

def make_id_dict(values):
    return {k:k for k in values}

def make_graph_agent(llm, vectorstore_ipcc, vectorstore_graphs, reranker, threshold_docs=0.2):
    
    workflow = StateGraph(GraphState)

    # Define the node functions
    categorize_intent = make_intent_categorization_node(llm)
    transform_query = make_query_transform_node(llm)
    translate_query = make_translation_node(llm)
    answer_chitchat = make_chitchat_node(llm)
    answer_ai_impact = make_ai_impact_node(llm)
    retrieve_documents = make_retriever_node(vectorstore_ipcc, reranker)
    retrieve_graphs = make_graph_retriever_node(vectorstore_graphs, reranker)
    answer_rag_graph = make_rag_graph_node(llm)
    answer_rag = make_rag_node(llm, with_docs=True)
    answer_rag_no_docs = make_rag_node(llm, with_docs=False)

    # Define the nodes
    workflow.add_node("set_defaults", set_defaults)
    workflow.add_node("categorize_intent", categorize_intent)
    workflow.add_node("search", search)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("translate_query", translate_query)
    workflow.add_node("answer_chitchat", answer_chitchat)
    workflow.add_node("answer_ai_impact", answer_ai_impact)
    workflow.add_node("retrieve_graphs", retrieve_graphs)
    workflow.add_node("answer_rag_graph", answer_rag_graph)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("answer_rag", answer_rag)
    workflow.add_node("answer_rag_no_docs", answer_rag_no_docs)

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
        "retrieve_documents",
        lambda x : route_based_on_relevant_docs(x,threshold_docs=threshold_docs),
        make_id_dict(["answer_rag","answer_rag_no_docs"])
    )

    # Define the edges
    workflow.add_edge("set_defaults", "categorize_intent")
    workflow.add_edge("translate_query", "transform_query")
    workflow.add_edge("transform_query", "retrieve_graphs")
    workflow.add_edge("retrieve_graphs", "answer_rag_graph")
    workflow.add_edge("answer_rag_graph", "retrieve_documents")
    workflow.add_edge("answer_rag", END)
    workflow.add_edge("answer_rag_no_docs", END)
    workflow.add_edge("answer_chitchat", END)
    workflow.add_edge("answer_ai_impact", END)

    # Compile
    app = workflow.compile()
    return app




def display_graph(app):

    display(
        Image(
            app.get_graph(xray = True).draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )

# import sys
# import os
# from contextlib import contextmanager

# from langchain.schema import Document
# from langgraph.graph import END, StateGraph
# from langchain_core.runnables.graph import CurveStyle, NodeColors, MermaidDrawMethod

# from typing_extensions import TypedDict
# from typing import List

# from IPython.display import display, HTML, Image

# from .chains.answer_chitchat import make_chitchat_node
# from .chains.answer_ai_impact import make_ai_impact_node
# from .chains.query_transformation import make_query_transform_node
# from .chains.translation import make_translation_node
# from .chains.intent_categorization import make_intent_categorization_node
# from .chains.retriever import make_retriever_node
# from .chains.answer_rag import make_rag_node


# class GraphState(TypedDict):
#     """
#     Represents the state of our graph.
#     """
#     user_input : str
#     language : str
#     intent : str
#     query: str
#     questions : List[dict]
#     answer: str
#     audience: str = "experts"
#     sources_input: List[str] = ["auto"]
#     documents: List[Document]

# def search(state):
#     return {}

# def route_intent(state):
#     intent = state["intent"]
#     if intent in ["chitchat","esg"]:
#         return "answer_chitchat"
#     elif intent == "ai_impact":
#         return "answer_ai_impact"
#     else:
#         # Search route
#         return "search"
    
# def route_translation(state):
#     if state["language"].lower() == "english":
#         return "transform_query"
#     else:
#         return "translate_query"
    
# def route_based_on_relevant_docs(state,threshold_docs=0.2):
#     docs = [x for x in state["documents"] if x.metadata["reranking_score"] > threshold_docs]
#     if len(docs) > 0:
#         return "answer_rag"
#     else:
#         return "answer_rag_no_docs"
    

# def make_id_dict(values):
#     return {k:k for k in values}

# def make_graph_agent(llm,vectorstore,reranker,threshold_docs = 0.2):
    
#     workflow = StateGraph(GraphState)

#     # Define the node functions
#     categorize_intent = make_intent_categorization_node(llm)
#     transform_query = make_query_transform_node(llm)
#     translate_query = make_translation_node(llm)
#     answer_chitchat = make_chitchat_node(llm)
#     answer_ai_impact = make_ai_impact_node(llm)
#     retrieve_documents = make_retriever_node(vectorstore,reranker)
#     answer_rag = make_rag_node(llm,with_docs=True)
#     answer_rag_no_docs = make_rag_node(llm,with_docs=False)

#     # Define the nodes
#     workflow.add_node("categorize_intent", categorize_intent)
#     workflow.add_node("search", search)
#     workflow.add_node("transform_query", transform_query)
#     workflow.add_node("translate_query", translate_query)
#     workflow.add_node("answer_chitchat", answer_chitchat)
#     workflow.add_node("answer_ai_impact", answer_ai_impact)
#     workflow.add_node("retrieve_documents",retrieve_documents)
#     workflow.add_node("answer_rag",answer_rag)
#     workflow.add_node("answer_rag_no_docs",answer_rag_no_docs)

#     # Entry point
#     workflow.set_entry_point("categorize_intent")

#     # CONDITIONAL EDGES
#     workflow.add_conditional_edges(
#         "categorize_intent",
#         route_intent,
#         make_id_dict(["answer_chitchat","answer_ai_impact","search"])
#     )

#     workflow.add_conditional_edges(
#         "search",
#         route_translation,
#         make_id_dict(["translate_query","transform_query"])
#     )

#     workflow.add_conditional_edges(
#         "retrieve_documents",
#         lambda x : route_based_on_relevant_docs(x,threshold_docs=threshold_docs),
#         make_id_dict(["answer_rag","answer_rag_no_docs"])
#     )

#     # Define the edges
#     workflow.add_edge("translate_query", "transform_query")
#     workflow.add_edge("transform_query", "retrieve_documents")
#     workflow.add_edge("retrieve_documents", "answer_rag")
#     workflow.add_edge("answer_rag", END)
#     workflow.add_edge("answer_rag_no_docs", END)
#     workflow.add_edge("answer_chitchat", END)
#     workflow.add_edge("answer_ai_impact", END)

#     # Compile
#     app = workflow.compile()
#     return app




# def display_graph(app):

#     display(
#         Image(
#             app.get_graph(xray = True).draw_mermaid_png(
#                 draw_method=MermaidDrawMethod.API,
#             )
#         )
#     )