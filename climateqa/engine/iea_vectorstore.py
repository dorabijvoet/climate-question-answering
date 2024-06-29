from langchain_chroma import Chroma

def get_chroma_vectorstore(embedding_function, persist_directory="/home/dora/climate-question-answering/data/vectorstore_cache"):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    return vectorstore