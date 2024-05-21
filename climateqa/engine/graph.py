import sys
import os
from contextlib import contextmanager

from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List

from .chains.answer_chitchat import make_chitchat_node
from .chains.answer_ai_impact import make_ai_impact_node
from .chains.query_transform import make_query_transform_node
from .chains.translation import make_translation_node
from .chains.intent_routing import make_intent_router_node


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
    sources_input: str
    documents: List[Document]

def search(state):
    return {}