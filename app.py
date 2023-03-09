import gradio as gr
from transformers import pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever
import numpy as np
import openai
import os


document_store = FAISSDocumentStore.load(
    index_path=f"./documents/climate_gpt.faiss",
    config_path=f"./documents/climate_gpt.json",
)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
system_template = {"role": os.environ["role"], "content": os.environ["content"]}

dense = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
)


def is_climate_change_related(sentence: str) -> bool:
    results = classifier(
        sequences=sentence,
        candidate_labels=["climate change related", "non climate change related"],
    )
    return results["labels"][np.argmax(results["scores"])] == "climate change related"


def make_pairs(lst):
    """from a list of even lenght, make tupple pairs"""
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]


def gen_conv(query: str, history=[system_template], ipcc=True):
    """return (answer:str, history:list[dict], sources:str)"""
    retrieve = ipcc and is_climate_change_related(query)
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
    messages.append({"role": "assistant", "content": answer})
    gradio_format = make_pairs([a["content"] for a in messages[1:]])

    return gradio_format, messages, sources


def set_openai_api_key(text):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    openai.api_key = os.environ["api_key"]

    if text.startswith("sk-") and len(text) > 10:
        openai.api_key = text
    return f"You're all set: this is your api key: {openai.api_key}"


# Gradio
with gr.Blocks(title="Eki IPCC Explorer") as demo:
    openai.api_key = os.environ["api_key"]
    gr.Markdown("# Climate GPT")
    # with gr.Row():
    #     gr.Markdown("First step: Add your OPENAI api key")
    #     openai_api_key_textbox = gr.Textbox(
    #         placeholder="Paste your OpenAI API key (sk-...) and hit Enter",
    #         show_label=False,
    #         lines=1,
    #         type="password",
    #     )

    gr.Markdown("""# Ask me anything, I'm a climate expert""")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            state = gr.State([system_template])

            with gr.Row():
                ask = gr.Textbox(
                    show_label=False, placeholder="Enter text and press enter"
                ).style(container=False)

        with gr.Column(scale=1, variant="panel"):

            gr.Markdown("### Sources")
            sources_textbox = gr.Textbox(
                interactive=False, show_label=False, max_lines=50
            )
    ask.submit(
        fn=gen_conv, inputs=[ask, state], outputs=[chatbot, state, sources_textbox]
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
