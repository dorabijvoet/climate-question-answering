from langchain_community.chat_models import AzureChatOpenAI
import os
# LOAD ENVIRONMENT VARIABLES
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


def get_llm(max_tokens = 1024,temperature = 0.0,verbose = True,streaming = False, **kwargs):

    llm = AzureChatOpenAI(
        openai_api_base=os.environ["AZURE_OPENAI_API_BASE_URL"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        deployment_name=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        openai_api_type = "azure",
        max_tokens = max_tokens,
        temperature = temperature,
        request_timeout = 60,
        verbose = verbose,
        streaming = streaming,
        **kwargs,
    )
    return llm
