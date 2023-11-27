# https://github.com/langchain-ai/langchain/issues/8623

import pandas as pd

from langchain.schema.retriever import BaseRetriever, Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores import VectorStore
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from pydantic import Field

class ClimateQARetriever(BaseRetriever):
    vectorstore:VectorStore
    sources:list = ["IPCC","IPBES"]
    reports:list = []
    threshold:float = 0.6
    k_summary:int = 3
    k_total:int = 10
    namespace:str = "vectors"


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # Check if all elements in the list are either IPCC or IPBES
        assert isinstance(self.sources,list)
        assert all([x in ["IPCC","IPBES"] for x in self.sources])
        assert self.k_total > self.k_summary, "k_total should be greater than k_summary"

        # Prepare base search kwargs

        filters = {}
        if len(self.reports) > 0:
            filters["short_name"] = {"$in":self.reports}
        else:
            filters["source"] = { "$in":self.sources}

        # Search for k_summary documents in the summaries dataset
        filters_summaries = {
            **filters,
            "report_type": { "$in":["SPM","TS"]},
        }

        docs_summaries = self.vectorstore.similarity_search_with_score(query=query,filter = filters_summaries,k = self.k_summary)
        docs_summaries = [x for x in docs_summaries if x[1] > self.threshold]

        # Search for k_total - k_summary documents in the full reports dataset
        filters_full = {
            **filters,
            "report_type": { "$nin":["SPM","TS"]},
        }
        k_full = self.k_total - len(docs_summaries)
        docs_full = self.vectorstore.similarity_search_with_score(query=query,filter = filters_full,k = k_full)

        # Concatenate documents
        docs = docs_summaries + docs_full

        # Filter if scores are below threshold
        docs = [x for x in docs if x[1] > self.threshold]

        # Add score to metadata
        results = []
        for i,(doc,score) in enumerate(docs):
            doc.metadata["similarity_score"] = score
            doc.metadata["content"] = doc.page_content
            doc.metadata["page_number"] = int(doc.metadata["page_number"])
            # doc.page_content = f"""Doc {i+1} - {doc.metadata['short_name']}: {doc.page_content}"""
            results.append(doc)

        # Sort by score
        # results = sorted(results,key = lambda x : x.metadata["similarity_score"],reverse = True)

        return results




# def filter_summaries(df,k_summary = 3,k_total = 10):
#     # assert source in ["IPCC","IPBES","ALL"], "source arg should be in (IPCC,IPBES,ALL)"

#     # # Filter by source
#     # if source == "IPCC":
#     #     df = df.loc[df["source"]=="IPCC"]
#     # elif source == "IPBES":
#     #     df = df.loc[df["source"]=="IPBES"]
#     # else:
#     #     pass

#     # Separate summaries and full reports
#     df_summaries = df.loc[df["report_type"].isin(["SPM","TS"])]
#     df_full = df.loc[~df["report_type"].isin(["SPM","TS"])]

#     # Find passages from summaries dataset
#     passages_summaries = df_summaries.head(k_summary)

#     # Find passages from full reports dataset
#     passages_fullreports = df_full.head(k_total - len(passages_summaries))

#     # Concatenate passages
#     passages = pd.concat([passages_summaries,passages_fullreports],axis = 0,ignore_index = True)
#     return passages




# def retrieve_with_summaries(query,retriever,k_summary = 3,k_total = 10,sources = ["IPCC","IPBES"],max_k = 100,threshold = 0.555,as_dict = True,min_length = 300):
#     assert max_k > k_total

#     validated_sources = ["IPCC","IPBES"]
#     sources = [x for x in sources if x in validated_sources]
#     filters = {
#         "source": { "$in": sources },
#     }
#     print(filters)

#     # Retrieve documents
#     docs = retriever.retrieve(query,top_k = max_k,filters = filters)

#     # Filter by score
#     docs = [{**x.meta,"score":x.score,"content":x.content} for x in docs if x.score > threshold]

#     if len(docs) == 0:
#         return []
#     res = pd.DataFrame(docs)
#     passages_df = filter_summaries(res,k_summary,k_total)
#     if as_dict:
#         contents = passages_df["content"].tolist()
#         meta = passages_df.drop(columns = ["content"]).to_dict(orient = "records")
#         passages = []
#         for i in range(len(contents)):
#             passages.append({"content":contents[i],"meta":meta[i]})
#         return passages
#     else:
#         return passages_df



# def retrieve(query,sources = ["IPCC"],threshold = 0.555,k = 10):


#     print("hellooooo")

#     # Reformulate queries
#     reformulated_query,language = reformulate(query)

#     print(reformulated_query)

#     # Retrieve documents
#     passages = retrieve_with_summaries(reformulated_query,retriever,k_total = k,k_summary = 3,as_dict = True,sources = sources,threshold = threshold)
#     response = {
#       "query":query,
#       "reformulated_query":reformulated_query,
#       "language":language,
#       "sources":passages,
#       "prompts":{"init_prompt":init_prompt,"sources_prompt":sources_prompt},
#     }
#     return response

