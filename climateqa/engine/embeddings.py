
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings_function(version = "v1.2"):

    if version == "v1.2":

        # https://huggingface.co/BAAI/bge-base-en-v1.5
        # Best embedding model at a reasonable size at the moment (2023-11-22)
    
        model_name = "BAAI/bge-base-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        embeddings_function = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
            query_instruction="Represent this sentence for searching relevant passages: "
        )

    else:

        embeddings_function = HuggingFaceEmbeddings(model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1")

    return embeddings_function