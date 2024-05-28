from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.base import format_document

from climateqa.engine.chains.prompts import answer_prompt_template,answer_prompt_without_docs_template,answer_prompt_images_template
from climateqa.engine.chains.prompts import papers_prompt_template

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, sep="\n\n"
):

    doc_strings =  []

    for i,doc in enumerate(docs):
        # chunk_type = "Doc" if doc.metadata["chunk_type"] == "text" else "Image"
        chunk_type = "Doc"
        if isinstance(doc,str):
            doc_formatted = doc
        else:
            doc_formatted = format_document(doc, document_prompt)
        doc_string = f"{chunk_type} {i+1}: " + doc_formatted
        doc_string = doc_string.replace("\n"," ") 
        doc_strings.append(doc_string)

    return sep.join(doc_strings)


def get_text_docs(x):
    return [doc for doc in x if doc.metadata["chunk_type"] == "text"]

def get_image_docs(x):
    return [doc for doc in x if doc.metadata["chunk_type"] == "image"]

def make_rag_chain(llm):
    prompt = ChatPromptTemplate.from_template(answer_prompt_template)
    chain = ({
        "context":lambda x : _combine_documents(x["documents"]),
        "query":itemgetter("query"),
        "language":itemgetter("language"),
        "audience":itemgetter("audience"),
    } | prompt | llm | StrOutputParser())
    return chain

def make_rag_chain_without_docs(llm):
    prompt = ChatPromptTemplate.from_template(answer_prompt_without_docs_template)
    chain = prompt | llm | StrOutputParser()
    return chain


def make_rag_node(llm,with_docs = True):

    if with_docs:
        rag_chain = make_rag_chain(llm)
    else:
        rag_chain = make_rag_chain_without_docs(llm)

    async def answer_rag(state,config):
        answer = await rag_chain.ainvoke(state,config)
        return {"answer":answer}

    return answer_rag




# def make_rag_papers_chain(llm):

#     prompt = ChatPromptTemplate.from_template(papers_prompt_template)
#     input_documents = {
#         "context":lambda x : _combine_documents(x["docs"]),
#         **pass_values(["question","language"])
#     }

#     chain = input_documents | prompt | llm | StrOutputParser()
#     chain = rename_chain(chain,"answer")

#     return chain






# def make_illustration_chain(llm):

#     prompt_with_images = ChatPromptTemplate.from_template(answer_prompt_images_template)

#     input_description_images = {
#         "images":lambda x : _combine_documents(get_image_docs(x["docs"])),
#         **pass_values(["question","audience","language","answer"]),
#     }

#     illustration_chain = input_description_images | prompt_with_images | llm | StrOutputParser()
#     return illustration_chain