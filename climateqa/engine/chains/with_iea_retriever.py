import sys
import os
from contextlib import contextmanager

from climateqa.engine.reranker import rerank_docs
from climateqa.engine.iea_retriever import IEARetriever
from climateqa.engine.retriever import ClimateQARetriever



def divide_into_parts(target, parts):
    # Base value for each part
    base = target // parts
    # Remainder to distribute
    remainder = target % parts
    # List to hold the result
    result = []
    
    for i in range(parts):
        if i < remainder:
            # These parts get base value + 1
            result.append(base + 1)
        else:
            # The rest get the base value
            result.append(base)
    
    return result


@contextmanager
def suppress_output():
    # Open a null device
    with open(os.devnull, 'w') as devnull:
        # Store the original stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # Redirect stdout and stderr to the null device
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr



def make_cqa_retriever_node(vectorstore,reranker,rerank_by_question=True, k_final=15, k_before_reranking=100, k_summary=5):

    def retrieve_documents_cqa(state):
        
        POSSIBLE_SOURCES = ["IPCC","IPBES","IPOS","OpenAlex"]
        questions = state["questions"]
        sources_input = state["sources_input"]
        
        # not necessary cuz fixed defaults
        # # Use sources from the user input or from the LLM detection
        # if "sources_input" not in state or state["sources_input"] is None:
        #     sources_input = ["auto"]
        # else:
        #     sources_input = state["sources_input"]
        # auto_mode = "auto" in sources_input

        auto_mode = "auto" in sources_input

        # There are several options to get the final top k
        # Option 1 - Get 100 documents by question and rerank by question
        # Option 2 - Get 100/n documents by question and rerank the total
        if rerank_by_question:
            k_by_question = divide_into_parts(k_final,len(questions))
        
        docs = []
        
        for i,q in enumerate(questions):
            
            sources = q["sources"]
            question = q["question"]
            
            # If auto mode, we use the sources detected by the LLM
            if auto_mode:
                sources = [x for x in sources if x in POSSIBLE_SOURCES]
                
            # Otherwise, we use the config
            else:
                sources = sources_input
                
            # Search the document store using the retriever
            # Configure high top k for further reranking step
            retriever = ClimateQARetriever(
                vectorstore = vectorstore,
                sources = sources,
                # reports = ias_reports,
                min_size = 200,
                k_summary = k_summary,
                k_total = k_before_reranking,
                threshold = 0.5,
                )
            docs_question = retriever.get_relevant_documents(question)
            
            # Rerank
            if reranker is not None:
                with suppress_output():
                    docs_question = rerank_docs(reranker,docs_question,question)
            else:
                # Add a default reranking score
                for doc in docs_question:
                    doc.metadata["reranking_score"] = doc.metadata["similarity_score"]
                
            # If rerank by question we select the top documents for each question
            if rerank_by_question:
                docs_question = docs_question[:k_by_question[i]]
                
            # Add sources used in the metadata
            for doc in docs_question:
                doc.metadata["sources_used"] = sources
            
            # Add to the list of docs
            docs.extend(docs_question)
            
        # Sorting the list in descending order by rerank_score
        # Then select the top k
        docs = sorted(docs, key=lambda x: x.metadata["reranking_score"], reverse=True)
        docs = docs[:k_final]
        
        new_state = {"documents":docs}

        return new_state
     
    return retrieve_documents_cqa


def make_iea_retriever_node(vectorstore,reranker,rerank_by_question=True, k_final=15, k_before_reranking=100):

        def retrieve_documents_IEA(state):
            
            POSSIBLE_SOURCES = ["IEA"] # plus tard ajouter OurWorldInData
            questions = state["questions"]
            sources_input = state["sources_input"]
            
            # not necessary cuz fixed defaults
            # # Use sources from the user input or from the LLM detection
            # if "sources_input" not in state or state["sources_input"] is None:
            #     sources_input = ["auto"]
            # else:
            #     sources_input = state["sources_input"]

            auto_mode = "auto" in sources_input

            # There are several options to get the final top k
            # Option 1 - Get 100 documents by question and rerank by question
            # Option 2 - Get 100/n documents by question and rerank the total
            if rerank_by_question:
                k_by_question = divide_into_parts(k_final,len(questions))
            
            docs = []
            
            for i,q in enumerate(questions):
                
                sources = q["sources"]
                question = q["question"]
                
                # If auto mode, we use the sources detected by the LLM
                if auto_mode:
                    sources = [x for x in sources if x in POSSIBLE_SOURCES]
                    
                # Otherwise, we use the config
                else:
                    sources = sources_input

                if any([x in POSSIBLE_SOURCES for x in sources]):

                    sources = [x for x in sources if x in POSSIBLE_SOURCES]
                    
                    # Search the document store using the retriever
                    # Configure high top k for further reranking step
                    retriever = IEARetriever(
                        vectorstore = vectorstore,
                        sources = sources,
                        # reports = ias_reports,
                        min_size = 5,
                        k_total = k_before_reranking,
                        threshold = 0.5,
                        )
                    docs_question = retriever.get_relevant_documents(question)

                    print(f" docs retrieved: {docs_question}")
                    
                    # Rerank
                    if reranker is not None:
                        with suppress_output():
                            docs_question = rerank_docs(reranker,docs_question,question)
                    else:
                        # Add a default reranking score
                        for doc in docs_question:
                            doc.metadata["reranking_score"] = doc.metadata["similarity_score"]
                        
                    # If rerank by question we select the top documents for each question
                    if rerank_by_question:
                        docs_question = docs_question[:k_by_question[i]]
                        
                    # Add sources used in the metadata
                    for doc in docs_question:
                        doc.metadata["sources_used"] = sources
                    
                    # Add to the list of docs
                    docs.extend(docs_question)
                    
                # Sorting the list in descending order by rerank_score
                # Then select the top k
                docs = sorted(docs, key=lambda x: x.metadata["reranking_score"], reverse=True)
                docs = docs[:k_final]
                
                new_state = {"recommended_content":docs}

            return new_state
        
        return retrieve_documents_IEA