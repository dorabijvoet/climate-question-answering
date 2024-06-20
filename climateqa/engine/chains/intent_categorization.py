
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


class IntentCategorizer(BaseModel):
    """Analyzing the user message input"""
    
    language: str = Field(
        description="Find the language of the message input in full words (ex: French, English, Spanish, ...), defaults to English",
        default="English",
    )
    intent: str = Field(
        enum=[
            "ai",
            "geo_info",
            "esg",
            "search",
            "chitchat",
        ],
        description="""
            Categorize the user input in one of the following category
            Any question

            Examples:
            - ai = Environmental impacts of AI: "What are the environmental impacts of AI", "How does AI affect the environment", "What is the carbon footprint of Artificial Intelligence", "How does AI contribute to climate change"
            - geo_info = Geolocated info about climate change: Any question where the user wants to know localized impacts of climate change, eg: "What will be the temperature in Marseille in 2050"
            - esg = Any question about the ESG regulation, frameworks and standards like the CSRD, TCFD, SASB, GRI, CDP, etc.
            - search = Searching for any quesiton about climate change, energy, biodiversity, nature, and everything we can find the IPCC or IPBES reports or scientific papers,
            - chitchat = Any general question that is not related to the environment or climate change or just conversational, or if you don't think searching the IPCC or IPBES reports would be relevant
        """,
    )



def make_intent_categorization_chain(llm):

    openai_functions = [convert_to_openai_function(IntentCategorizer)]
    llm_with_functions = llm.bind(functions = openai_functions,function_call={"name":"IntentCategorizer"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant, you will analyze, translate and reformulate the user input message using the function provided"),
        ("user", "input: {input}")
    ])

    chain = prompt | llm_with_functions | JsonOutputFunctionsParser()
    return chain


def make_intent_categorization_node(llm):

    categorization_chain = make_intent_categorization_chain(llm)

    def categorize_message(state):
        output = categorization_chain.invoke({"input":state["user_input"]})
        if "language" not in output: output["language"] = "English"
        output["query"] = state["user_input"]
        return output
    
    return categorize_message




# SAMPLE_QUESTIONS = [
#     "Est-ce que l'IA a un impact sur l'environnement ?",
#     "Que dit le GIEC sur l'impact de l'IA",
#     "Qui sont les membres du GIEC",
#     "What is the impact of El Nino ?",
#     "Yo",
#     "Hello ça va bien ?",
#     "Par qui as tu été créé ?",
#     "What role do cloud formations play in modulating the Earth's radiative balance, and how are they represented in current climate models?",
#     "Which industries have the highest GHG emissions?",
#     "What are invasive alien species and how do they threaten biodiversity and ecosystems?",
#     "Are human activities causing global warming?",
#     "What is the motivation behind mining the deep seabed?",
#     "Tu peux m'écrire un poème sur le changement climatique ?",
#     "Tu peux m'écrire un poème sur les bonbons ?",
#     "What will be the temperature in 2100 in Strasbourg?",
#     "C'est quoi le lien entre biodiversity and changement climatique ?",
# ]