
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
    base_path = "search_demo/climateq&a/processed_image/"
    path = os.path.join(base_path, path)
    file_object = get_file_from_azure_blob_storage(path)
    image = Image.open(file_object)
    return image
