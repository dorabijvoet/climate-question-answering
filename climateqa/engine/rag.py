from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document

from climateqa.engine.reformulation import make_reformulation_chain
from climateqa.engine.prompts import answer_prompt_template
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
    answer = {
        "answer": input_documents | prompt | llm | StrOutputParser(),
        **pass_values(["question","audience","language","query","docs"])
    }

    # ------- FINAL CHAIN
    # Build the final chain
    rag_chain = reformulation | find_documents | answer

    return rag_chain

