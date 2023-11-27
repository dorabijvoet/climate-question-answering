
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from climateqa.engine.prompts import reformulation_prompt_template



response_schemas = [
    ResponseSchema(name="language", description="The detected language of the input message"),
    ResponseSchema(name="question", description="The reformulated question always in English")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


def make_reformulation_chain(llm):

    prompt = PromptTemplate(
        template=reformulation_prompt_template,
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions}
    )

    chain = (prompt | llm.bind(stop=["```"]) | output_parser)
    return chain
