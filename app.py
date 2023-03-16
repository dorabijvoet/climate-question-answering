import gradio as gr
from transformers import pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
import numpy as np
import openai
import os
from datasets import load_dataset
from datasets import Dataset
import time
from utils import is_climate_change_related, make_pairs, set_openai_api_key


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
system_template = {"role": os.environ["role"], "content": os.environ["content"]}


def gen_conv(query: str, report_type, history=[system_template], ipcc=True):
    """return (answer:str, history:list[dict], sources:str)

    Args:
        query (str): _description_
        history (list, optional): _description_. Defaults to [system_template].
        ipcc (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if report_type == "IPCC only":
        document_store = FAISSDocumentStore.load(
            index_path="./documents/climate_gpt_only_giec.faiss",
            config_path="./documents/climate_gpt_only_giec.json",
        )
    else:
        document_store = FAISSDocumentStore.load(
            index_path="./documents/climate_gpt.faiss",
            config_path="./documents/climate_gpt.json",
        )

    dense = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers",
    )

    retrieve = ipcc and is_climate_change_related(query, classifier)

    sources = ""
    messages = history + [
        {"role": "user", "content": query},
    ]

    if retrieve:
        docs = dense.retrieve(query=query, top_k=5)
        sources = "\n\n".join(
            [os.environ["sources"]]
            + [
                f"{d.meta['file_name']} Page {d.meta['page_number']}\n{d.content}"
                for d in docs
            ]
        )
        messages.append({"role": "system", "content": sources})

    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        #         max_tokens=200,
    )["choices"][0]["message"]["content"]

    if retrieve:
        messages.pop()
        # answer = "(top 5 documents retrieved) " + answer
        sources = "\n\n".join(
            f"{d.meta['file_name']} Page {d.meta['page_number']}:\n{d.content}"
            for d in docs
        )
    else:
        sources = "No environmental report was used to provide this answer."

    messages.append({"role": "assistant", "content": answer})
    gradio_format = make_pairs([a["content"] for a in messages[1:]])

    return gradio_format, messages, sources


# Gradio
css_code = ".gradio-container {background-image: url('file=background.png');background-position: top right}"

with gr.Blocks(title="🌍 ClimateGPT Ekimetrics", css=css_code) as demo:

    openai.api_key = os.environ["api_key"]
    gr.Markdown("### Welcome to Climate GPT 🌍 ! ")
    gr.Markdown(
        """
        Climate GPT is an interactive exploration tool designed to help you easily find relevant information based on  of Environmental reports such as IPCCs and ??.

        IPCC is a United Nations body that assesses the science related to climate change, including its impacts and possible response options. The IPCC is considered the leading scientific authority on all things related to global climate change.
    """
    )
    gr.Markdown(
        "**How does it work:** This Chatbot is a combination of two technologies. FAISS search applied to a vast amount of scientific climate reports and TurboGPT to generate human-like text from the part of the document extracted from the database."
    )
    gr.Markdown(
        "⚠️ Warning: Always refer to the source (on the right side) to ensure the validity of the information communicated"
    )
    # gr.Markdown("""### Ask me anything, I'm a climate expert""")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
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
            sources_textbox = gr.Textbox(
                interactive=False, show_label=False, max_lines=50
            )

    ask.submit(
        fn=gen_conv,
        inputs=[
            ask,
            gr.inputs.Dropdown(
                ["IPCC only", "All available"],
                default="All available",
                label="Select reports",
            ),
            state,
        ],
        outputs=[chatbot, state, sources_textbox],
    )
    with gr.Accordion("Add your personal openai api key", open=False):
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...) and hit Enter",
            show_label=False,
            lines=1,
            type="password",
        )
    openai_api_key_textbox.change(set_openai_api_key, inputs=[openai_api_key_textbox])
    openai_api_key_textbox.submit(set_openai_api_key, inputs=[openai_api_key_textbox])

demo.launch()
