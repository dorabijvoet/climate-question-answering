from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.base import format_document

from climateqa.engine.reformulation import make_reformulation_chain
from climateqa.engine.prompts import answer_prompt_template,answer_prompt_without_docs_template
from climateqa.engine.utils import pass_values, flatten_dict


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, sep="\n\n"
):
    doc_strings = [f"Doc {i+1}: " + format_document(doc, document_prompt) for i,doc in enumerate(docs)]
    return sep.join(doc_strings)


def make_rag_chain(retriever,llm):


    # Construct the prompt
    prompt = ChatPromptTemplate.from_template(answer_prompt_template)
    prompt_without_docs = ChatPromptTemplate.from_template(answer_prompt_without_docs_template)

    # ------- CHAIN 0 - Reformulation
    reformulation_chain = make_reformulation_chain(llm)
    reformulation = (
        {"reformulation":reformulation_chain,**pass_values(["audience","query"])}
        | RunnablePassthrough()
        | flatten_dict
    )


    # ------- CHAIN 1
    # Retrieved documents
    find_documents =  {
        "docs": itemgetter("question") | retriever,
        **pass_values(["question","audience","language","query"])
    } | RunnablePassthrough()


    # ------- CHAIN 2
    # Construct inputs for the llm
    input_documents = {
        "context":lambda x : _combine_documents(x["docs"]),
        **pass_values(["question","audience","language"])
    }

    # Generate the answer



    answer_with_docs = {
        "answer": input_documents | prompt | llm | StrOutputParser(),
        **pass_values(["question","audience","language","query","docs"])
    }

    answer_without_docs = {
        "answer":  prompt_without_docs | llm | StrOutputParser(),
        **pass_values(["question","audience","language","query","docs"])
    }

    answer = RunnableBranch(
        (lambda x: len(x["docs"]) > 0, answer_with_docs),
        answer_without_docs,
    )


    # ------- FINAL CHAIN
    # Build the final chain
    rag_chain = reformulation | find_documents | answer

    return rag_chain

