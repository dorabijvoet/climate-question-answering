from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.base import format_document

from climateqa.engine.chains.prompts import answer_prompt_template,answer_prompt_without_docs_template,answer_prompt_images_template, answer_prompt_iea_template
from climateqa.engine.chains.prompts import papers_prompt_template

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="""Graph title: {page_content}. URL of the graph: {url}""")

def _combine_recommended_content(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, sep="\n\n"
):

    doc_strings =  []

    for i,doc in enumerate(docs):
        # chunk_type = "Doc" if doc.metadata["chunk_type"] == "text" else "Image"
        chunk_type = "Graph"
        if isinstance(doc,str):
            doc_formatted = doc
        else:
            doc_formatted = format_document(doc, document_prompt)

        doc_string = f"{chunk_type} {i+1}: " + doc_formatted
        # doc_string = doc_string.replace("\n"," ") 
        doc_strings.append(doc_string)

    return sep.join(doc_strings)


def get_text_docs(x):
    return [doc for doc in x if doc.metadata["chunk_type"] == "text"]

def get_image_docs(x):
    return [doc for doc in x if doc.metadata["chunk_type"] == "image"]

def make_rag_chain_with_recommended_content(llm):
    prompt = ChatPromptTemplate.from_template(answer_prompt_iea_template)
    chain = ({
        "recommended_content":lambda x : _combine_recommended_content(x["recommended_content"]),
        "answer":itemgetter("answer"),
        "query":itemgetter("query"),
        "language":itemgetter("language"),
        "audience":itemgetter("audience"),
    } | prompt | llm | StrOutputParser())
    return chain


def make_rag_node_recommended_content(llm, with_docs=True):
    if with_docs:
        rag_chain = make_rag_chain_with_recommended_content(llm)
    else:
        return {}

    async def answer_rag(state, config):
        answer = await rag_chain.ainvoke(state, config)

        state["recommended_content_answer"] = answer

        return state
    
    # Dora
    # async def answer_rag(state, config):
    #     answer = await rag_chain.ainvoke(state, config)

    #     answer = f"{state['answer']} \n\n\n {answer}"

    #     state["recommended_content_answer"] = answer
    #     state["answer"] = answer

    #     return state

    return answer_rag