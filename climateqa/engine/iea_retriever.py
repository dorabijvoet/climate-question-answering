from langchain_core.retrievers import BaseRetriever
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from typing import List
from pydantic import Field

class IEARetriever(BaseRetriever):
    vectorstore:VectorStore
    sources:list = ["IEA"] # plus tard ajouter OurWorldInData # faudra integrate avec l'autre retriever
    reports:list = [] # example: Global Critical Mineral Outlook 2024 (appears_in in charts.csv)
    threshold:float = 0.5
    k_total:int = 10
    namespace:str = "vectors"
    min_size:int = 5 # titles of graphs are very short

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # Check if all elements in the list are IEA (later also OurWorldInData)
        assert isinstance(self.sources,list)
        assert all([x in ["IEA"] for x in self.sources])

        # Prepare base search kwargs
        filters = {}

        if len(self.reports) > 0:
            filters["appears_in"] = {"$in": self.reports}
        else:
            filters["source"] = {"$in": self.sources}

        docs = self.vectorstore.similarity_search_with_score(query=query, filter=filters, k=self.k_total)
        
        # Filter if scores are below threshold or min size
        docs = [x for x in docs if len(x[0].page_content) > self.min_size]
        docs = [x for x in docs if x[1] > self.threshold]

        # Add score to metadata
        results = []
        for i,(doc,score) in enumerate(docs):
            doc.metadata["similarity_score"] = score
            doc.metadata["content"] = doc.page_content
            results.append(doc)

        return results