# Pinecone
# More info at https://docs.pinecone.io/docs/langchain
# And https://python.langchain.com/docs/integrations/vectorstores/pinecone
import os
import pinecone
from langchain.vectorstores import Pinecone

# LOAD ENVIRONMENT VARIABLES
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


def get_pinecone_vectorstore(embeddings,text_key = "text"):

    # initialize pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_API_ENVIRONMENT"),  # next to api key in console
    )

    index_name = os.getenv("PINECONE_API_INDEX")
    vectorstore = Pinecone.from_existing_index(index_name, embeddings,text_key = text_key)
    return vectorstore


# def get_pinecone_retriever(vectorstore,k = 10,namespace = "vectors",sources = ["IPBES","IPCC"]):

#     assert isinstance(sources,list)

#     # Check if all elements in the list are either IPCC or IPBES
#     filter = {
#         "source": { "$in":sources},
#     }

#     retriever = vectorstore.as_retriever(search_kwargs={
#         "k": k,
#         "namespace":"vectors",
#         "filter":filter
#     })

#     return retriever