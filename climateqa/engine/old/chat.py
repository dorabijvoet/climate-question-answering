# LANGCHAIN IMPORTS
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


# CLIMATEQA
from climateqa.retriever import ClimateQARetriever
from climateqa.vectorstore import get_pinecone_vectorstore
from climateqa.chains import load_climateqa_chain


class ClimateQA:
    def __init__(self,hf_embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                 show_progress_bar = False,batch_size = 1,max_tokens = 1024,**kwargs):
    
        self.llm = self.get_llm(max_tokens = max_tokens,**kwargs)
        self.embeddings_function = HuggingFaceEmbeddings(
            model_name=hf_embedding_model,
            encode_kwargs={"show_progress_bar":show_progress_bar,"batch_size":batch_size}
        )



    def get_vectorstore(self):
        pass


    def reformulate(self):
        pass


    def retrieve(self):
        pass


    def ask(self):
        pass