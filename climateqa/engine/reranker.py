import os
from dotenv import load_dotenv
from scipy.special import expit, logit
from rerankers import Reranker
from sentence_transformers import CrossEncoder

load_dotenv()

def get_reranker(model = "jina", cohere_api_key = None):
    
    assert model in ["nano","tiny","small","large", "jina"]

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
    elif model == "jina":
        # Reached token quota so does not work
        reranker = Reranker("jina-reranker-v2-base-multilingual", api_key = os.getenv("JINA_RERANKER_API_KEY"))
        # marche pas sans gpu ? et anyways returns with another structure donc faudrait changer le code du retriever node
        # reranker = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", automodel_args={"torch_dtype": "auto"}, trust_remote_code=True,)
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