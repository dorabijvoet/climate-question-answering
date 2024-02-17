import os
import time

from langchain_openai import AzureChatOpenAI
from msal import ConfidentialClientApplication

DEFAULT_TOKEN_UPDATE_FREQUENCY = 3300  # Default token duration is 1 hour (3600 s.)

# LOAD ENVIRONMENT VARIABLES
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


client_id = os.environ.get("AZURE_CLIENT_ID", None)
client_credential = os.environ.get("AZURE_CLIENT_CREDENTIAL", None)
tenant_name = os.environ.get("AZURE_TENANT_NAME", None)
scopes = [os.environ.get("AZURE_SCOPE", None)]

azure_ad_token_frequency = int(
    os.environ.get("TOKEN_UPDATE_FREQUENCY", DEFAULT_TOKEN_UPDATE_FREQUENCY)
)
azure_ad_token = None
azure_ad_token_timestamp = 0.0


def _get_azure_ad_token():
    global azure_ad_token
    global azure_ad_token_timestamp
    now = time.time()

    # Return current token if not outdated:
    if (azure_ad_token is not None) and (
        azure_ad_token_timestamp + azure_ad_token_frequency > now
    ):
        print("Using current token (not expired)...")
        return azure_ad_token

    # Else, generate a new token:
    print("Generating new token...")
    app = ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_credential,
        authority=f"https://login.microsoftonline.com/{tenant_name}",
    )
    result = app.acquire_token_for_client(scopes=scopes)
    if "access_token" not in result:
        raise ValueError("No access token in result")

    if result["access_token"] != azure_ad_token:
        print("New token received.")
        azure_ad_token = result["access_token"]
        azure_ad_token_timestamp = now
    else:
        print("Same token received.")

    return azure_ad_token


def get_llm(
    max_tokens: int = 1024,
    temperature: float = 0.0,
    verbose: bool = True,
    streaming: bool = False,
    **kwargs,
) -> AzureChatOpenAI:
    auth_dict = dict(openai_api_type="azure")
    # Note: OPENAI_API_VERSION is automatically taken from environment variables.

    # First option: provide AZURE_OPENAI_API_BASE_URL, OPENAI_API_VERSION, AZURE_CLIENT_ID,
    # AZURE_CLIENT_CREDENTIAL, AZURE_TENANT_NAME & AZURE_SCOPE:
    if (
        (client_id is not None)
        and (client_credential is not None)
        and (tenant_name is not None)
    ):
        print("Using Azure AD token")
        auth_dict["openai_api_base"] = os.environ["AZURE_OPENAI_API_BASE_URL"]
        auth_dict["azure_ad_token_provider"] = _get_azure_ad_token

    # Second option: provide AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_DEPLOYMENT_NAME,
    # OPENAI_API_VERSION & AZURE_OPENAI_API_KEY:
    else:
        print("Using AZURE_OPENAI_API_DEPLOYMENT_NAME and AZURE_OPENAI_API_KEY")
        auth_dict["deployment_name"] = os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME"]
        # Note: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are automatically taken
        # from environment variable.

    llm = AzureChatOpenAI(
        **auth_dict,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=verbose,
        streaming=streaming,
        **kwargs,
    )
    return llm
