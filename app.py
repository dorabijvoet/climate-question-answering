from climateqa.engine.embeddings import get_embeddings_function
embeddings_function = get_embeddings_function()


import gradio as gr
import pandas as pd
import numpy as np
import os
import time
import re
import json

# from gradio_modal import Modal

from io import BytesIO
import base64

from datetime import datetime
from azure.storage.fileshare import ShareServiceClient

from utils import create_user_id



# ClimateQ&A imports
from climateqa.engine.llm import get_llm
from climateqa.engine.rag import make_rag_chain
from climateqa.engine.vectorstore import get_pinecone_vectorstore
from climateqa.engine.retriever import ClimateQARetriever
from climateqa.engine.embeddings import get_embeddings_function
from climateqa.engine.prompts import audience_prompts
from climateqa.sample_questions import QUESTIONS
from climateqa.constants import POSSIBLE_REPORTS
from climateqa.utils import get_image_from_azure_blob_storage

# Load environment variables in local mode
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    pass

# Set up Gradio Theme
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="red",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
)



init_prompt = ""

system_template = {
    "role": "system",
    "content": init_prompt,
}

account_key = os.environ["BLOB_ACCOUNT_KEY"]
if len(account_key) == 86:
    account_key += "=="

credential = {
    "account_key": account_key,
    "account_name": os.environ["BLOB_ACCOUNT_NAME"],
}

account_url = os.environ["BLOB_ACCOUNT_URL"]
file_share_name = "climateqa"
service = ShareServiceClient(account_url=account_url, credential=credential)
share_client = service.get_share_client(file_share_name)

user_id = create_user_id()



def parse_output_llm_with_sources(output):
    # Split the content into a list of text and "[Doc X]" references
    content_parts = re.split(r'\[(Doc\s?\d+(?:,\s?Doc\s?\d+)*)\]', output)
    parts = []
    for part in content_parts:
        if part.startswith("Doc"):
            subparts = part.split(",")
            subparts = [subpart.lower().replace("doc","").strip() for subpart in subparts]
            subparts = [f"<span class='doc-ref'><sup>{subpart}</sup></span>" for subpart in subparts]
            parts.append("".join(subparts))
        else:
            parts.append(part)
    content_parts = "".join(parts)
    return content_parts


# Create vectorstore and retriever
vectorstore = get_pinecone_vectorstore(embeddings_function)
llm = get_llm(provider="openai",max_tokens = 1024,temperature = 0.0)


def make_pairs(lst):
    """from a list of even lenght, make tupple pairs"""
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]


def serialize_docs(docs):
    new_docs = []
    for doc in docs:
        new_doc = {}
        new_doc["page_content"] = doc.page_content
        new_doc["metadata"] = doc.metadata
        new_docs.append(new_doc)
    return new_docs



async def chat(query,history,audience,sources,reports):
    """taking a query and a message history, use a pipeline (reformulation, retriever, answering) to yield a tuple of:
    (messages in gradio format, messages in langchain format, source documents)"""

    print(f">> NEW QUESTION : {query}")

    if audience == "Children":
        audience_prompt = audience_prompts["children"]
    elif audience == "General public":
        audience_prompt = audience_prompts["general"]
    elif audience == "Experts":
        audience_prompt = audience_prompts["experts"]
    else:
        audience_prompt = audience_prompts["experts"]

    # Prepare default values
    if len(sources) == 0:
        sources = ["IPCC"]

    if len(reports) == 0:
        reports = []

    retriever = ClimateQARetriever(vectorstore=vectorstore,sources = sources,min_size = 200,reports = reports,k_summary = 3,k_total = 15,threshold=0.5)
    rag_chain = make_rag_chain(retriever,llm)
    
    inputs = {"query": query,"audience": audience_prompt}
    result = rag_chain.astream_log(inputs) #{"callbacks":[MyCustomAsyncHandler()]})
    # result = rag_chain.stream(inputs)

    path_reformulation = "/logs/reformulation/final_output"
    path_retriever = "/logs/find_documents/final_output"
    path_answer = "/logs/answer/streamed_output_str/-"

    docs_html = ""
    output_query = ""
    output_language = ""
    gallery = []

    try:
        async for op in result:

            op = op.ops[0]
            # print("ITERATION",op)

            if op['path'] == path_reformulation: # reforulated question
                try:
                    output_language = op['value']["language"] # str
                    output_query = op["value"]["question"]
                except Exception as e:
                    raise gr.Error(f"ClimateQ&A Error: {e} - The error has been noted, try another question and if the error remains, you can contact us :)")
            
            elif op['path'] == path_retriever: # documents
                try:
                    docs = op['value']['docs'] # List[Document]
                    docs_html = []
                    for i, d in enumerate(docs, 1):
                        docs_html.append(make_html_source(d, i))
                    docs_html = "".join(docs_html)
                except TypeError:
                    print("No documents found")
                    print("op: ",op)
                    continue

            elif op['path'] == path_answer: # final answer
                new_token = op['value'] # str
                # time.sleep(0.01)
                previous_answer = history[-1][1]
                previous_answer = previous_answer if previous_answer is not None else ""
                answer_yet = previous_answer + new_token
                answer_yet = parse_output_llm_with_sources(answer_yet)
                history[-1] = (query,answer_yet)

        
            # elif op['path'] == final_output_path_id:
            #     final_output = op['value']

            #     if "answer" in final_output:
                
            #         final_output = final_output["answer"]
            #         print(final_output)
            #         answer = history[-1][1] + final_output
            #         answer = parse_output_llm_with_sources(answer)
            #         history[-1] = (query,answer)

            else:
                continue

            history = [tuple(x) for x in history]
            yield history,docs_html,output_query,output_language,gallery

    except Exception as e:
        raise gr.Error(f"{e}")


    try:
        # Log answer on Azure Blob Storage
        if os.getenv("GRADIO_ENV") != "local":
            timestamp = str(datetime.now().timestamp())
            file = timestamp + ".json"
            prompt = history[-1][0]
            logs = {
                "user_id": str(user_id),
                "prompt": prompt,
                "query": prompt,
                "question":output_query,
                "sources":sources,
                "docs":serialize_docs(docs),
                "answer": history[-1][1],
                "time": timestamp,
            }
            log_on_azure(file, logs, share_client)
    except Exception as e:
        print(f"Error logging on Azure Blob Storage: {e}")
        raise gr.Error(f"ClimateQ&A Error: {str(e)[:100]} - The error has been noted, try another question and if the error remains, you can contact us :)")

    image_dict = {}
    for i,doc in enumerate(docs):
        
        if doc.metadata["chunk_type"] == "image":
            try:
                key = f"Image {i+1}"
                image_path = doc.metadata["image_path"].split("documents/")[1]
                img = get_image_from_azure_blob_storage(image_path)

                # Convert the image to a byte buffer
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Embedding the base64 string in Markdown
                markdown_image = f"![Alt text](data:image/png;base64,{img_str})"
                image_dict[key] = {"img":img,"md":markdown_image,"caption":doc.page_content,"key":key,"figure_code":doc.metadata["figure_code"]}
            except Exception as e:
                print(f"Skipped adding image {i} because of {e}")

    if len(image_dict) > 0:

        gallery = [x["img"] for x in list(image_dict.values())]
        img = list(image_dict.values())[0]
        img_md = img["md"]
        img_caption = img["caption"]
        img_code = img["figure_code"]
        if img_code != "N/A":
            img_name = f"{img['key']} - {img['figure_code']}"
        else:
            img_name = f"{img['key']}"

        answer_yet = history[-1][1] + f"\n\n{img_md}\n<p class='chatbot-caption'><b>{img_name}</b> - {img_caption}</p>"
        history[-1] = (history[-1][0],answer_yet)
        history = [tuple(x) for x in history]

    # gallery = [x.metadata["image_path"] for x in docs if (len(x.metadata["image_path"]) > 0 and "IAS" in x.metadata["image_path"])]
    # if len(gallery) > 0:
    #     gallery = list(set("|".join(gallery).split("|")))
    #     gallery = [get_image_from_azure_blob_storage(x) for x in gallery]

    yield history,docs_html,output_query,output_language,gallery


    # memory.save_context(inputs, {"answer": gradio_format[-1][1]})
    # yield gradio_format, memory.load_memory_variables({})["history"], source_string
    
# async def chat_with_timeout(query, history, audience, sources, reports, timeout_seconds=2):
#     async def timeout_gen(async_gen, timeout):
#         try:
#             while True:
#                 try:
#                     yield await asyncio.wait_for(async_gen.__anext__(), timeout)
#                 except StopAsyncIteration:
#                     break
#         except asyncio.TimeoutError:
#             raise gr.Error("Operation timed out. Please try again.")

#     return timeout_gen(chat(query, history, audience, sources, reports), timeout_seconds)



# # A wrapper function that includes a timeout
# async def chat_with_timeout(query, history, audience, sources, reports, timeout_seconds=2):
#     try:
#         # Use asyncio.wait_for to apply a timeout to the chat function
#         return await asyncio.wait_for(chat(query, history, audience, sources, reports), timeout_seconds)
#     except asyncio.TimeoutError:
#         # Handle the timeout error as desired
#         raise gr.Error("Operation timed out. Please try again.")




def make_html_source(source,i):
    meta = source.metadata
    # content = source.page_content.split(":",1)[1].strip()
    content = source.page_content.strip()

    toc_levels = []
    for j in range(2):
        level = meta[f"toc_level{j}"]
        if level != "N/A":
            toc_levels.append(level)
        else:
            break
    toc_levels = " > ".join(toc_levels)

    if len(toc_levels) > 0:
        name = f"<b>{toc_levels}</b><br/>{meta['name']}"
    else:
        name = meta['name']

    if meta["chunk_type"] == "text":

        card = f"""
    <div class="card">
        <div class="card-content">
            <h2>Doc {i} - {meta['short_name']} - Page {int(meta['page_number'])}</h2>
            <p>{content}</p>
        </div>
        <div class="card-footer">
            <span>{name}</span>
            <a href="{meta['url']}#page={int(meta['page_number'])}" target="_blank" class="pdf-link">
                <span role="img" aria-label="Open PDF">🔗</span>
            </a>
        </div>
    </div>
    """
    
    else:

        if meta["figure_code"] != "N/A":
            title = f"{meta['figure_code']} - {meta['short_name']}"
        else:
            title = f"{meta['short_name']}"

        card = f"""
    <div class="card card-image">
        <div class="card-content">
            <h2>Image {i} - {title} - Page {int(meta['page_number'])}</h2>
            <p>{content}</p>
            <p class='ai-generated'>AI-generated description</p>
        </div>
        <div class="card-footer">
            <span>{name}</span>
            <a href="{meta['url']}#page={int(meta['page_number'])}" target="_blank" class="pdf-link">
                <span role="img" aria-label="Open PDF">🔗</span>
            </a>
        </div>
    </div>
    """
        
    return card



#     else:
#         docs_string = "No relevant passages found in the climate science reports (IPCC and IPBES)"
#         complete_response = "**No relevant passages found in the climate science reports (IPCC and IPBES), you may want to ask a more specific question (specifying your question on climate issues).**"
#         messages.append({"role": "assistant", "content": complete_response})
#         gradio_format = make_pairs([a["content"] for a in messages[1:]])
#         yield gradio_format, messages, docs_string


def save_feedback(feed: str, user_id):
    if len(feed) > 1:
        timestamp = str(datetime.now().timestamp())
        file = user_id + timestamp + ".json"
        logs = {
            "user_id": user_id,
            "feedback": feed,
            "time": timestamp,
        }
        log_on_azure(file, logs, share_client)
        return "Feedback submitted, thank you!"




def log_on_azure(file, logs, share_client):
    logs = json.dumps(logs)
    file_client = share_client.get_file_client(file)
    file_client.upload_file(logs)


# --------------------------------------------------------------------
# Gradio
# --------------------------------------------------------------------


init_prompt = """
Hello, I am ClimateQ&A, a conversational assistant designed to help you understand climate change and biodiversity loss. I will answer your questions by **sifting through the IPCC and IPBES scientific reports**.

❓ How to use
- **Language**: You can ask me your questions in any language. 
- **Audience**: You can specify your audience (children, general public, experts) to get a more adapted answer.
- **Sources**: You can choose to search in the IPCC or IPBES reports, or both.

⚠️ Limitations
*Please note that the AI is not perfect and may sometimes give irrelevant answers. If you are not satisfied with the answer, please ask a more specific question or report your feedback to help us improve the system.*

What do you want to learn ?
"""


def vote(data: gr.LikeData):
    if data.liked:
        print(data.value)
    else:
        print(data)



with gr.Blocks(title="Climate Q&A", css="style.css", theme=theme,elem_id = "main-component") as demo:
    # user_id_state = gr.State([user_id])

    with gr.Tab("ClimateQ&A"):

        with gr.Row(elem_id="chatbot-row"):
            with gr.Column(scale=2):
                # state = gr.State([system_template])
                chatbot = gr.Chatbot(
                    value=[(None,init_prompt)],
                    show_copy_button=True,show_label = False,elem_id="chatbot",layout = "panel",
                    avatar_images = (None,"https://i.ibb.co/YNyd5W2/logo4.png"),
                )#,avatar_images = ("assets/logo4.png",None))
                
                # bot.like(vote,None,None)



                with gr.Row(elem_id = "input-message"):
                    textbox=gr.Textbox(placeholder="Ask me anything here!",show_label=False,scale=7,lines = 1,interactive = True,elem_id="input-textbox")
                    # submit = gr.Button("",elem_id = "submit-button",scale = 1,interactive = True,icon = "https://static-00.iconduck.com/assets.00/settings-icon-2048x2046-cw28eevx.png")


            with gr.Column(scale=1, variant="panel",elem_id = "right-panel"):


                with gr.Tabs() as tabs:
                    with gr.TabItem("Examples",elem_id = "tab-examples",id = 0):
                                        
                        examples_hidden = gr.Textbox(visible = False)
                        first_key = list(QUESTIONS.keys())[0]
                        dropdown_samples = gr.Dropdown(QUESTIONS.keys(),value = first_key,interactive = True,show_label = True,label = "Select a category of sample questions",elem_id = "dropdown-samples")

                        samples = []
                        for i,key in enumerate(QUESTIONS.keys()):

                            examples_visible = True if i == 0 else False

                            with gr.Row(visible = examples_visible) as group_examples:

                                examples_questions = gr.Examples(
                                    QUESTIONS[key],
                                    [examples_hidden],
                                    examples_per_page=8,
                                    run_on_click=False,
                                    elem_id=f"examples{i}",
                                    api_name=f"examples{i}",
                                    # label = "Click on the example question or enter your own",
                                    # cache_examples=True,
                                )
                            
                            samples.append(group_examples)


                    with gr.Tab("Citations",elem_id = "tab-citations",id = 1):
                        sources_textbox = gr.HTML(show_label=False, elem_id="sources-textbox")
                        docs_textbox = gr.State("")

                    # with Modal(visible = False) as config_modal:
                    with gr.Tab("Configuration",elem_id = "tab-config",id = 2):

                        gr.Markdown("Reminder: You can talk in any language, ClimateQ&A is multi-lingual!")


                        dropdown_sources = gr.CheckboxGroup(
                            ["IPCC", "IPBES","IPOS"],
                            label="Select source",
                            value=["IPCC"],
                            interactive=True,
                        )

                        dropdown_reports = gr.Dropdown(
                            POSSIBLE_REPORTS,
                            label="Or select specific reports",
                            multiselect=True,
                            value=None,
                            interactive=True,
                        )

                        dropdown_audience = gr.Dropdown(
                            ["Children","General public","Experts"],
                            label="Select audience",
                            value="Experts",
                            interactive=True,
                        )

                        output_query = gr.Textbox(label="Query used for retrieval",show_label = True,elem_id = "reformulated-query",lines = 2,interactive = False)
                        output_language = gr.Textbox(label="Language",show_label = True,elem_id = "language",lines = 1,interactive = False)





#---------------------------------------------------------------------------------------
# OTHER TABS
#---------------------------------------------------------------------------------------


    with gr.Tab("Figures",elem_id = "tab-images",elem_classes = "max-height other-tabs"):
        gallery_component = gr.Gallery()

    with gr.Tab("About",elem_classes = "max-height other-tabs"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("See more info at [https://climateqa.com](https://climateqa.com/docs/intro/)")


    def start_chat(query,history):
        history = history + [(query,None)]
        history = [tuple(x) for x in history]
        return (gr.update(interactive = False),gr.update(selected=1),history)
    
    def finish_chat():
        return (gr.update(interactive = True,value = ""))

    (textbox
        .submit(start_chat, [textbox,chatbot], [textbox,tabs,chatbot],queue = False,api_name = "start_chat_textbox")
        .then(chat, [textbox,chatbot,dropdown_audience, dropdown_sources,dropdown_reports], [chatbot,sources_textbox,output_query,output_language,gallery_component],concurrency_limit = 8,api_name = "chat_textbox")
        .then(finish_chat, None, [textbox],api_name = "finish_chat_textbox")
    )

    (examples_hidden
        .change(start_chat, [examples_hidden,chatbot], [textbox,tabs,chatbot],queue = False,api_name = "start_chat_examples")
        .then(chat, [examples_hidden,chatbot,dropdown_audience, dropdown_sources,dropdown_reports], [chatbot,sources_textbox,output_query,output_language,gallery_component],concurrency_limit = 8,api_name = "chat_examples")
        .then(finish_chat, None, [textbox],api_name = "finish_chat_examples")
    )


    def change_sample_questions(key):
        index = list(QUESTIONS.keys()).index(key)
        visible_bools = [False] * len(samples)
        visible_bools[index] = True
        return [gr.update(visible=visible_bools[i]) for i in range(len(samples))]



    dropdown_samples.change(change_sample_questions,dropdown_samples,samples)

    # # textbox.submit(predict_climateqa,[textbox,bot],[None,bot,sources_textbox])
    # (textbox
    #     .submit(answer_user, [textbox,examples_hidden, bot], [textbox, bot],queue = False)
    #     .success(change_tab,None,tabs)
    #     .success(fetch_sources,[textbox,dropdown_sources], [textbox,sources_textbox,docs_textbox,output_query,output_language])
    #     .success(answer_bot, [textbox,bot,docs_textbox,output_query,output_language,dropdown_audience], [textbox,bot],queue = True)
    #     .success(lambda x : textbox,[textbox],[textbox])
    # )

    # (examples_hidden
    #     .change(answer_user_example, [textbox,examples_hidden, bot], [textbox, bot],queue = False)
    #     .success(change_tab,None,tabs)
    #     .success(fetch_sources,[textbox,dropdown_sources], [textbox,sources_textbox,docs_textbox,output_query,output_language])
    #     .success(answer_bot, [textbox,bot,docs_textbox,output_query,output_language,dropdown_audience], [textbox,bot],queue=True)
    #     .success(lambda x : textbox,[textbox],[textbox])
    # )
    # submit_button.click(answer_user, [textbox, bot], [textbox, bot], queue=True).then(
    #         answer_bot, [textbox,bot,dropdown_audience,dropdown_sources], [textbox,bot,sources_textbox]
    #     )


    # with Modal(visible=True) as first_modal:
    #     gr.Markdown("# Welcome to ClimateQ&A !")

    #     gr.Markdown("### Examples")

    #     examples = gr.Examples(
    #         ["Yo ça roule","ça boume"],
    #         [examples_hidden],
    #         examples_per_page=8,
    #         run_on_click=False,
    #         elem_id="examples",
    #         api_name="examples",
    #     )


    # submit.click(lambda: Modal(visible=True), None, config_modal)
    

    demo.queue()

demo.launch()
