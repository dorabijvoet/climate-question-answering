
from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch

from climateqa.engine.chains.prompts import reformulation_prompt_template
from climateqa.engine.utils import pass_values, flatten_dict


response_schemas = [
    ResponseSchema(name="language", description="The detected language of the input message"),
    ResponseSchema(name="question", description="The reformulated question always in English")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

def fallback_default_values(x):
    if x["question"] is None:
        x["question"] = x["query"]
        x["language"] = "english"
    
    return x

def make_reformulation_chain(llm):

    prompt = PromptTemplate(
        template=reformulation_prompt_template,
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions}
    )

    chain = (prompt | llm.bind(stop=["```"]) | output_parser)

    reformulation_chain = (
        {"reformulation":chain,**pass_values(["query"])}
        | RunnablePassthrough()
        | flatten_dict
        | fallback_default_values
    )


    return reformulation_chain
