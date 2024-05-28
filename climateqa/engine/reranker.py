import os
from scipy.special import expit, logit
from rerankers import Reranker


def get_reranker(model = "nano",cohere_api_key = None):
    
    assert model in ["nano","tiny","small","large"]

    if model == "nano":
        reranker = Reranker('ms-marco-TinyBERT-L-2-v2', model_type='flashrank')
    elif model == "tiny":
        reranker = Reranker('ms-marco-MiniLM-L-12-v2', model_type='flashrank')
    elif model == "small":
        reranker = Reranker("mixedbread-ai/mxbai-rerank-xsmall-v1", model_type='cross-encoder')
    elif model == "large":
        if cohere_api_key is None:
            cohere_api_key = os.environ["COHERE_API_KEY"]
        reranker = Reranker("cohere", lang='en', api_key = cohere_api_key)
    return reranker



def rerank_docs(reranker,docs,query):
    
    # Get a list of texts from langchain docs
    input_docs = [x.page_content for x in docs]
    
    # Rerank using rerankers library
    results = reranker.rank(query=query, docs=input_docs)

    # Prepare langchain list of docs
    docs_reranked = []
    for result in results.results:
        doc_id = result.document.doc_id
        doc = docs[doc_id]
        doc.metadata["reranking_score"] = result.score
        doc.metadata["query_used_for_retrieval"] = query
        docs_reranked.append(doc)
    return docs_reranked