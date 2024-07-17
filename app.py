from climateqa.engine.embeddings import get_embeddings_function
embeddings_function = get_embeddings_function()

from climateqa.papers.openalex import OpenAlex
from sentence_transformers import CrossEncoder

# reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
oa = OpenAlex()

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
from climateqa.engine.vectorstore import get_pinecone_vectorstore
from climateqa.engine.retriever import ClimateQARetriever
from climateqa.engine.reranker import get_reranker
from climateqa.engine.embeddings import get_embeddings_function
from climateqa.engine.chains.prompts import audience_prompts
from climateqa.sample_questions import QUESTIONS
from climateqa.constants import POSSIBLE_REPORTS
from climateqa.utils import get_image_from_azure_blob_storage
from climateqa.engine.keywords import make_keywords_chain
# from climateqa.engine.chains.answer_rag import make_rag_papers_chain
from climateqa.engine.graph import make_graph_agent,display_graph

from front.utils import make_html_source,parse_output_llm_with_sources,serialize_docs,make_toolbox

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



# Create vectorstore and retriever
vectorstore = get_pinecone_vectorstore(embeddings_function)
llm = get_llm(provider="openai",max_tokens = 1024,temperature = 0.0)
reranker = get_reranker("nano")
agent = make_graph_agent(llm,vectorstore,reranker)




async def chat(query,history,audience,sources,reports):
    """taking a query and a message history, use a pipeline (reformulation, retriever, answering) to yield a tuple of:
    (messages in gradio format, messages in langchain format, source documents)"""

    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f">> NEW QUESTION ({date_now}) : {query}")

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
    
    inputs = {"user_input": query,"audience": audience_prompt,"sources":sources}
    result = agent.astream_events(inputs,version = "v1") #{"callbacks":[MyCustomAsyncHandler()]})
    # result = rag_chain.stream(inputs)

    # path_reformulation = "/logs/reformulation/final_output"
    # path_keywords = "/logs/keywords/final_output"
    # path_retriever = "/logs/find_documents/final_output"
    # path_answer = "/logs/answer/streamed_output_str/-"

    docs = []
    docs_html = ""
    output_query = ""
    output_language = ""
    output_keywords = ""
    gallery = []
    start_streaming = False

    steps_display = {
        "categorize_intent":("🔄️ Analyzing user message",True),
        "transform_query":("🔄️ Thinking step by step to answer the question",True),
        "retrieve_documents":("🔄️ Searching in the knowledge base",False),
    }

    try:
        async for event in result:

            if event["event"] == "on_chat_model_stream":
                if start_streaming == False:
                    start_streaming = True
                    history[-1] = (query,"")

                new_token = event["data"]["chunk"].content
                # time.sleep(0.01)
                previous_answer = history[-1][1]
                previous_answer = previous_answer if previous_answer is not None else ""
                answer_yet = previous_answer + new_token
                answer_yet = parse_output_llm_with_sources(answer_yet)
                history[-1] = (query,answer_yet)

            
            elif event["name"] == "retrieve_documents" and event["event"] == "on_chain_end":
                try:
                    docs = event["data"]["output"]["documents"]
                    docs_html = []
                    for i, d in enumerate(docs, 1):
                        docs_html.append(make_html_source(d, i))
                    docs_html = "".join(docs_html)
                except Exception as e:
                    print(f"Error getting documents: {e}")
                    print(event)

            # elif event["name"] == "retrieve_documents" and event["event"] == "on_chain_start":
            #     print(event)
            #     questions = event["data"]["input"]["questions"]
            #     questions = "\n".join([f"{i+1}. {q['question']} ({q['source']})" for i,q in enumerate(questions)])
            #     answer_yet = "🔄️ Searching in the knowledge base\n{questions}"
            #     history[-1] = (query,answer_yet)


            for event_name,(event_description,display_output) in steps_display.items():
                if event["name"] == event_name:
                    if event["event"] == "on_chain_start":
                        # answer_yet = f"<p><span class='loader'></span>{event_description}</p>"
                        # answer_yet = make_toolbox(event_description, "", checked = False)
                        answer_yet = event_description
                        history[-1] = (query,answer_yet)
                    # elif event["event"] == "on_chain_end":
                    #     answer_yet = ""
                    #     history[-1] = (query,answer_yet)
                        # if display_output:
                        #     print(event["data"]["output"])

            # if op['path'] == path_reformulation: # reforulated question
            #     try:
            #         output_language = op['value']["language"] # str
            #         output_query = op["value"]["question"]
            #     except Exception as e:
            #         raise gr.Error(f"ClimateQ&A Error: {e} - The error has been noted, try another question and if the error remains, you can contact us :)")
            
            # if op["path"] == path_keywords:
            #     try:
            #         output_keywords = op['value']["keywords"] # str
            #         output_keywords = " AND ".join(output_keywords)
            #     except Exception as e:
            #         pass



            history = [tuple(x) for x in history]
            yield history,docs_html,output_query,output_language,gallery,output_query,output_keywords

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

    yield history,docs_html,output_query,output_language,gallery,output_query,output_keywords



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


def generate_keywords(query):
    chain = make_keywords_chain(llm)
    keywords = chain.invoke(query)
    keywords = " AND ".join(keywords["keywords"])
    return keywords



papers_cols_widths = {
    "doc":50,
    "id":100,
    "title":300,
    "doi":100,
    "publication_year":100,
    "abstract":500,
    "rerank_score":100,
    "is_oa":50,
}

papers_cols = list(papers_cols_widths.keys())
papers_cols_widths = list(papers_cols_widths.values())

# async def find_papers(query, keywords,after):

#     summary = ""
    
#     df_works = oa.search(keywords,after = after)
#     df_works = df_works.dropna(subset=["abstract"])
#     df_works = oa.rerank(query,df_works,reranker)
#     df_works = df_works.sort_values("rerank_score",ascending=False)
#     G = oa.make_network(df_works)

#     height = "750px"
#     network = oa.show_network(G,color_by = "rerank_score",notebook=False,height = height)
#     network_html = network.generate_html()

#     network_html = network_html.replace("'", "\"")
#     css_to_inject = "<style>#mynetwork { border: none !important; } .card { border: none !important; }</style>"
#     network_html = network_html + css_to_inject

    
#     network_html = f"""<iframe style="width: 100%; height: {height};margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
#     display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
#     allow-scripts allow-same-origin allow-popups 
#     allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
#     allowpaymentrequest="" frameborder="0" srcdoc='{network_html}'></iframe>"""


#     docs = df_works["content"].head(15).tolist()

#     df_works = df_works.reset_index(drop = True).reset_index().rename(columns = {"index":"doc"})
#     df_works["doc"] = df_works["doc"] + 1
#     df_works = df_works[papers_cols]

#     yield df_works,network_html,summary

#     chain = make_rag_papers_chain(llm)
#     result = chain.astream_log({"question": query,"docs": docs,"language":"English"})
#     path_answer = "/logs/StrOutputParser/streamed_output/-"

#     async for op in result:

#         op = op.ops[0]

#         if op['path'] == path_answer: # reforulated question
#             new_token = op['value'] # str
#             summary += new_token
#         else:
#             continue
#         yield df_works,network_html,summary
    


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


                    with gr.Tab("Sources",elem_id = "tab-citations",id = 1):
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

    # with gr.Tab("Papers (beta)",elem_id = "tab-papers",elem_classes = "max-height other-tabs"):

    #     with gr.Row():
    #         with gr.Column(scale=1):
    #             query_papers = gr.Textbox(placeholder="Question",show_label=False,lines = 1,interactive = True,elem_id="query-papers")
    #             keywords_papers = gr.Textbox(placeholder="Keywords",show_label=False,lines = 1,interactive = True,elem_id="keywords-papers")
    #             after = gr.Slider(minimum=1950,maximum=2023,step=1,value=1960,label="Publication date",show_label=True,interactive=True,elem_id="date-papers")
    #             search_papers = gr.Button("Search",elem_id="search-papers",interactive=True)

    #         with gr.Column(scale=7):

    #             with gr.Tab("Summary",elem_id="papers-summary-tab"):
    #                 papers_summary = gr.Markdown(visible=True,elem_id="papers-summary")

    #             with gr.Tab("Relevant papers",elem_id="papers-results-tab"):
    #                 papers_dataframe = gr.Dataframe(visible=True,elem_id="papers-table",headers = papers_cols)

    #             with gr.Tab("Citations network",elem_id="papers-network-tab"):
    #                 citations_network = gr.HTML(visible=True,elem_id="papers-citations-network")


            
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

    # query_papers.submit(generate_keywords,[query_papers], [keywords_papers])
    # search_papers.click(find_papers,[query_papers,keywords_papers,after], [papers_dataframe,citations_network,papers_summary])

    demo.queue()

demo.launch()
