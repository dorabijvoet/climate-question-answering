from langchain_openai import ChatOpenAI
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def get_llm(model="gpt-3.5-turbo-0125",max_tokens=1024, temperature=0.0, streaming=True,timeout=30, **kwargs):
    
    llm = ChatOpenAI(
        model=model,
        api_key=os.environ.get("THEO_API_KEY", None),
        max_tokens = max_tokens,
        streaming = streaming,
        temperature=temperature,
        timeout = timeout,
        **kwargs,
    )

    return llm