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


def chat(query: str, history: list = [system_template], report_type="All available", threshold=0.56):
    retriever = retrieve_all if report_type == "All available" else retrieve_giec
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
        messages.append({"role": "system", "content": "no relevant document available."})
        sources = "No environmental report was used to provide this answer."

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.2,)["choices"][0][
        "message"
    ]["content"]

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
            \n **How does it work:** This Chatbot is a combination of two technologies. FAISS search applied to a vast amount of scientific climate reports and TurboGPT to generate human-like text from the part of the document extracted from the database. 
            \n ⚠️ Warning: Always refer to the source to ensure the validity of the information communicated.
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
