
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings_function(version = "v1.2",query_instruction = "Represent this sentence for searching relevant passages: "):

    if version == "v1.2":

        # https://huggingface.co/BAAI/bge-base-en-v1.5
        # Best embedding model at a reasonable size at the moment (2023-11-22)
    
        model_name = "BAAI/bge-base-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True,"show_progress_bar":False} # set True to compute cosine similarity
        print("Loading embeddings model: ", model_name)
        embeddings_function = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
            query_instruction=query_instruction,
        )

    else:

        embeddings_function = HuggingFaceEmbeddings(model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1")

    return embeddings_function



