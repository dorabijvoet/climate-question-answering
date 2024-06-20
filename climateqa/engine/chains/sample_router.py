
# from typing import List
# from typing import Literal
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.utils.function_calling import convert_to_openai_function
# from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

# # https://livingdatalab.com/posts/2023-11-05-openai-function-calling-with-langchain.html

# class Location(BaseModel):
#     country:str = Field(...,description="The country if directly mentioned or inferred from the location (cities, regions, adresses), ex: France, USA, ...")
#     location:str = Field(...,description="The specific place if mentioned (cities, regions, addresses), ex: Marseille, New York, Wisconsin, ...")

# class QueryAnalysis(BaseModel):
#     """Analyzing the user query"""
    
#     language: str = Field(
#         description="Find the language of the query in full words (ex: French, English, Spanish, ...), defaults to English"
#     )
#     intent: str = Field(
#         enum=[
#             "Environmental impacts of AI",
#             "Geolocated info about climate change",
#             "Climate change",
#             "Biodiversity",
#             "Deep sea mining",
#             "Chitchat",
#         ],
#         description="""
#             Categorize the user query in one of the following category, 

#             Examples:
#             - Geolocated info about climate change: "What will be the temperature in Marseille in 2050"
#             - Climate change: "What is radiative forcing", "How much will
#         """,
#     )
#     sources: List[Literal["IPCC", "IPBES", "IPOS"]] = Field(
#         ...,
#         description="""
#             Given a user question choose which documents would be most relevant for answering their question,
#             - IPCC is for questions about climate change, energy, impacts, and everything we can find the IPCC reports
#             - IPBES is for questions about biodiversity and nature
#             - IPOS is for questions about the ocean and deep sea mining
            
#         """,
#     )
#     date: str = Field(description="The date or period mentioned, ex: 2050, between 2020 and 2050")
#     location:Location
#     # query: str = Field(
#     #     description = """
#     #         Translate to english and reformulate the following user message to be a short standalone question, in the context of an educational discussion about climate change.
#     #         The reformulated question will used in a search engine
#     #         By default, assume that the user is asking information about the last century,
#     #         Use the following examples
            
#     #         ### Examples:
#     #         La technologie nous sauvera-t-elle ? -> Can technology help humanity mitigate the effects of climate change?
#     #         what are our reserves in fossil fuel? -> What are the current reserves of fossil fuels and how long will they last?
#     #         what are the main causes of climate change? -> What are the main causes of climate change in the last century?

#     #         Question in English:
#     #     """
#     # )

# openai_functions = [convert_to_openai_function(QueryAnalysis)]
# llm2 = llm.bind(functions = openai_functions,function_call={"name":"QueryAnalysis"})