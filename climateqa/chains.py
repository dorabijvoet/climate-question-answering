# https://python.langchain.com/docs/modules/chains/how_to/custom_chain
# Including reformulation of the question in the chain
import json

from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import TransformChain, SequentialChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from climateqa.prompts import answer_prompt, reformulation_prompt,audience_prompts


def load_reformulation_chain(llm):

    prompt = PromptTemplate(
        template = reformulation_prompt,
        input_variables=["query"],
    )
    reformulation_chain = LLMChain(llm = llm,prompt = prompt,output_key="json")

    # Parse the output
    def parse_output(output):
        query = output["query"]
        json_output = json.loads(output["json"])
        question = json_output.get("question", query)
        language = json_output.get("language", "English")
        return {
            "question": question,
            "language": language,
        }
    
    transform_chain = TransformChain(
        input_variables=["json"], output_variables=["question","language"], transform=parse_output
    )

    reformulation_chain = SequentialChain(chains = [reformulation_chain,transform_chain],input_variables=["query"],output_variables=["question","language"])
    return reformulation_chain



def load_answer_chain(retriever,llm):
    prompt = PromptTemplate(template=answer_prompt, input_variables=["summaries", "question","audience","language"])
    qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff",prompt = prompt)

    # This could be improved by providing a document prompt to avoid modifying page_content in the docs
    # See here https://github.com/langchain-ai/langchain/issues/3523

    answer_chain = RetrievalQAWithSourcesChain(
        combine_documents_chain = qa_chain,
        retriever=retriever,
        return_source_documents = True,
    )
    return answer_chain


def load_climateqa_chain(retriever,llm):

    reformulation_chain = load_reformulation_chain(llm)
    answer_chain = load_answer_chain(retriever,llm)

    climateqa_chain = SequentialChain(
        chains = [reformulation_chain,answer_chain],
        input_variables=["query","audience"],
        output_variables=["answer","question","language","source_documents"],
        return_all = True,
    )
    return climateqa_chain

