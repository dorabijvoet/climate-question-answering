from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.base import format_document

from climateqa.engine.reformulation import make_reformulation_chain
from climateqa.engine.prompts import answer_prompt_template,answer_prompt_without_docs_template,answer_prompt_images_template
from climateqa.engine.prompts import papers_prompt_template
from climateqa.engine.utils import pass_values, flatten_dict,prepare_chain,rename_chain
from climateqa.engine.keywords import make_keywords_chain

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


def make_rag_chain(retriever,llm):

    # Construct the prompt
    prompt = ChatPromptTemplate.from_template(answer_prompt_template)
    prompt_without_docs = ChatPromptTemplate.from_template(answer_prompt_without_docs_template)

    # ------- CHAIN 0 - Reformulation
    reformulation = make_reformulation_chain(llm)
    reformulation = prepare_chain(reformulation,"reformulation")

    # ------- Find all keywords from the reformulated query
    keywords = make_keywords_chain(llm)
    keywords = {"keywords":itemgetter("question") | keywords}
    keywords = prepare_chain(keywords,"keywords")

    # ------- CHAIN 1
    # Retrieved documents
    find_documents = {"docs": itemgetter("question") | retriever} | RunnablePassthrough()
    find_documents = prepare_chain(find_documents,"find_documents")

    # ------- CHAIN 2
    # Construct inputs for the llm
    input_documents = {
        "context":lambda x : _combine_documents(x["docs"]),
        **pass_values(["question","audience","language","keywords"])
    }

    # ------- CHAIN 3
    # Bot answer
    llm_final = rename_chain(llm,"answer")

    answer_with_docs = {
        "answer": input_documents | prompt | llm_final | StrOutputParser(),
        **pass_values(["question","audience","language","query","docs","keywords"]),
    }

    answer_without_docs = {
        "answer":  prompt_without_docs | llm_final | StrOutputParser(),
        **pass_values(["question","audience","language","query","docs","keywords"]),
    }

    # def has_images(x):
    #     image_docs = [doc for doc in x["docs"] if doc.metadata["chunk_type"]=="image"]
    #     return len(image_docs) > 0
    
    def has_docs(x):
        return len(x["docs"]) > 0

    answer = RunnableBranch(
        (lambda x: has_docs(x), answer_with_docs),
        answer_without_docs,
    )


    # ------- FINAL CHAIN
    # Build the final chain
    rag_chain = reformulation | keywords | find_documents | answer

    return rag_chain


def make_rag_papers_chain(llm):

    prompt = ChatPromptTemplate.from_template(papers_prompt_template)

    input_documents = {
        "context":lambda x : _combine_documents(x["docs"]),
        **pass_values(["question","language"])
    }

    chain = input_documents | prompt | llm | StrOutputParser()
    chain = rename_chain(chain,"answer")

    return chain






def make_illustration_chain(llm):

    prompt_with_images = ChatPromptTemplate.from_template(answer_prompt_images_template)

    input_description_images = {
        "images":lambda x : _combine_documents(get_image_docs(x["docs"])),
        **pass_values(["question","audience","language","answer"]),
    }

    illustration_chain = input_description_images | prompt_with_images | llm | StrOutputParser()
    return illustration_chain