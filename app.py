import gradio as gr
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
import openai
import os
from utils import (
    make_pairs,
    set_openai_api_key,
    get_random_string,
)

system_template = {"role": "system", "content": os.environ["content"]}

retrieve_all = EmbeddingRetriever(
    document_store=FAISSDocumentStore.load(
        index_path="./documents/climate_gpt.faiss",
        config_path="./documents/climate_gpt.json",
    ),
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
)
retrieve_giec = EmbeddingRetriever(
    document_store=FAISSDocumentStore.load(
        index_path="./documents/climate_gpt_only_giec.faiss",
        config_path="./documents/climate_gpt_only_giec.json",
    ),
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
)


def chat(
    query: str, history: list = [system_template], report_type: str = "All available", threshold: float = 0.559
) -> tuple:
    """retrieve relevant documents in the document store then query gpt-turbo

    Args:
        query (str): user message.
        history (list, optional): history of the conversation. Defaults to [system_template].
        report_type (str, optional): should be "All available" or "IPCC only". Defaults to "All available".
        threshold (float, optional): similarity threshold, don't increase more than 0.568. Defaults to 0.56.

    Yields:
        tuple: chat gradio format, chat openai format, sources used.
    """

    if report_type == "All available":
        retriever = retrieve_all
    elif report_type == "IPCC only":
        retriever = retrieve_giec
    else:
        raise Exception("report_type arg should be in (All available, IPCC only)")

    docs = retriever.retrieve(query=query, top_k=10)

    messages = history + [{"role": "user", "content": query}]
    sources = "\n\n".join(
        f"doc {i}: {d.meta['file_name']} page {d.meta['page_number']}\n{d.content}"
        for i, d in enumerate(docs, 1)
        if d.score > threshold
    )

    if sources:
        messages.append({"role": "system", "content": f"{os.environ['sources']}\n\n{sources}"})
    else:
        messages.append(
            {
                "role": "system",
                "content": """Tell the user you did not find any relevant documents. Answer using your knowledge. Then tell them if their message was too vague or too short and if relevant provide an example of an alternative query.
        example: 
        user: forrest
        assistant: I did not find any relevant documents to answer your question, I will answer using my knowledge as a language model.
        Forests play a crucial role in mitigating climate change. They act as carbon sinks, absorbing carbon dioxide from the atmosphere through photosynthesis and storing it in their biomass and soil. Deforestation, on the other hand, releases carbon dioxide into the atmosphere and reduces the planet's capacity to absorb carbon. Therefore, protecting and restoring forests is an important strategy for mitigating climate change.
        
        You may want to try one of those more specific questions:
        - What is deforestation, and how does it contribute to climate change?
        - How does climate change impact forests and their ecosystems?""",
            }
        )
        sources = "No environmental report was used to provide this answer."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    complete_response = ""
    for chunk in response:
        if chunk_message := chunk["choices"][0]["delta"].get("content", None):
            complete_response += chunk_message
            messages[-1] = {"role": "assistant", "content": complete_response}
            gradio_format = make_pairs([a["content"] for a in messages[1:]])
            yield gradio_format, messages, sources


def test(feed: str):
    print(feed)


def reset_textbox():
    return gr.update(value="")


# Gradio
css_code = ".gradio-container {background-image: url('file=background.png');background-position: top right}"

with gr.Blocks(title="🌍 ClimateGPT Ekimetrics", css=css_code) as demo:

    openai.api_key = os.environ["api_key"]
    user_id = gr.State([get_random_string(10)])

    with gr.Tab("App"):
        gr.Markdown("# Welcome to Climate GPT 🌍 !")
        gr.Markdown(
            """ Climate GPT is an interactive exploration tool designed to help you easily find relevant information based on  of Environmental reports such as IPCCs and other environmental reports.
            \n **How does it work:** when a user sends a message, the system retrieves the most relevant paragraphs from scientific reports that are semantically related to the user's question. These paragraphs are then used to generate a comprehensive and well-sourced answer using a language model.
            \n **Usage guideline:** more sources will be retrieved using precise questions
                instead of "forrest", you may want to try one of those more specific questions
                - How do forests help mitigate climate change?
                - What is deforestation, and how does it contribute to climate change?
                - How does climate change impact forests and their ecosystems?
                - Can reforestation efforts help combat climate change?
                - What role do forests play in the carbon cycle, and how does that impact climate change? 
            \n ⚠️ Always refer to the source to ensure the validity of the information communicated.
            """
        )
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(elem_id="chatbot")
                state = gr.State([system_template])

                with gr.Row():
                    ask = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                        sample_inputs=["which country polutes the most ?"],
                    ).style(container=False)
                    print(f"Type from ask textbox {ask.type}")

            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### Sources")
                sources_textbox = gr.Textbox(interactive=False, show_label=False, max_lines=50)

        ask.submit(
            fn=chat,
            inputs=[
                ask,
                state,
                gr.inputs.Dropdown(
                    ["IPCC only", "All available"],
                    default="All available",
                    label="Select reports",
                ),
            ],
            outputs=[chatbot, state, sources_textbox],
        )
        ask.submit(reset_textbox, [], [ask])

        with gr.Accordion("Feedbacks", open=False):
            gr.Markdown("Please complete some feedbacks 🙏")
            feedback = gr.Textbox()
            feedback_save = gr.Button(value="submit feedback")
            feedback_save.click(test, inputs=[feedback])

        with gr.Accordion("Add your personal openai api key - Option", open=False):
            openai_api_key_textbox = gr.Textbox(
                placeholder="Paste your OpenAI API key (sk-...) and hit Enter",
                show_label=False,
                lines=1,
                type="password",
            )
        openai_api_key_textbox.change(set_openai_api_key, inputs=[openai_api_key_textbox])
        openai_api_key_textbox.submit(set_openai_api_key, inputs=[openai_api_key_textbox])

    with gr.Tab("Information"):
        gr.Markdown(
            """
        ## 📖 Reports used : \n
        - First Assessment Report on the Physical Science of Climate Change
        - Second assessment Report on Climate Change Adaptation
        - Third Assessment Report on Climate Change Mitigation
        - Food Outlook Biannual Report on Global Food Markets
        - IEA's report on the Role of Critical Minerals in Clean Energy Transitions
        - Limits to Growth
        - Outside The Safe operating system of the Planetary Boundary for Novel Entities
        - Planetary Boundaries Guiding
        - State of the Oceans report
        - Word Energy Outlook 2021
        - Word Energy Outlook 2022
        - The environmental impacts of plastics and micro plastics use, waste and polution ET=U and national measures
        - IPBES Global report - MArch 2022

        \n
        IPCC is a United Nations body that assesses the science related to climate change, including its impacts and possible response options. 
        The IPCC is considered the leading scientific authority on all things related to global climate change.

        """
        )
    with gr.Tab("Examples"):
        gr.Markdown("See here some examples on how to use the Chatbot")

    demo.queue(concurrency_count=16)

demo.launch()
