from climateqa.engine.llm.openai import get_llm as get_openai_llm
from climateqa.engine.llm.azure import get_llm as get_azure_llm


def get_llm(provider="openai",**kwargs):

    if provider == "openai":
        return get_openai_llm(**kwargs)
    elif provider == "azure":
        return get_azure_llm(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    

