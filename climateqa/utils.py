
import io
import os
from PIL import Image
from azure.storage.blob import ContainerClient


def get_file_from_azure_blob_storage(path):
    AZURE_SAS_URL_TRD = os.environ["AZURE_SAS_URL_TRD"]
    container_client = ContainerClient.from_container_url(AZURE_SAS_URL_TRD)
    blob_client = container_client.get_blob_client(path)
    stream = blob_client.download_blob().readall()
    file_object = io.BytesIO(stream)
    return file_object


def get_image_from_azure_blob_storage(path):
    base_path = "climateqa/documents/"
    path = os.path.join(base_path, path)
    file_object = get_file_from_azure_blob_storage(path)
    image = Image.open(file_object)
    return image

def remove_duplicates_keep_highest_score(documents):
    unique_docs = {}
    
    for doc in documents:
        doc_id = doc.metadata.get('doc_id')
        if doc_id in unique_docs:
            if doc.metadata['reranking_score'] > unique_docs[doc_id].metadata['reranking_score']:
                unique_docs[doc_id] = doc
        else:
            unique_docs[doc_id] = doc
    
    return list(unique_docs.values())
