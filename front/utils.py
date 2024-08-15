
import re

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



def parse_output_llm_with_sources(output):
    # Split the content into a list of text and "[Doc X]" references
    content_parts = re.split(r'\[(Doc\s?\d+(?:,\s?Doc\s?\d+)*)\]', output)
    parts = []
    for part in content_parts:
        if part.startswith("Doc"):
            subparts = part.split(",")
            subparts = [subpart.lower().replace("doc","").strip() for subpart in subparts]
            subparts = [f"""<a href="#doc{subpart}" class="a-doc-ref" target="_self"><span class='doc-ref'><sup>{subpart}</sup></span></a>""" for subpart in subparts]
            parts.append("".join(subparts))
        else:
            parts.append(part)
    content_parts = "".join(parts)
    return content_parts


from collections import defaultdict

def generate_html_graphs(graphs):
    # Organize graphs by category
    categories = defaultdict(list)
    for graph in graphs:
        category = graph['metadata']['category']
        categories[category].append(graph['embedding'])

    # Begin constructing the HTML
    html_code = '''
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Graphs by Category</title>
                    <style>
                        .tab-content {
                            display: none;
                        }
                        .tab-content.active {
                            display: block;
                        }
                        .tabs {
                            margin-bottom: 20px;
                        }
                        .tab-button {
                            background-color: #ddd;
                            border: none;
                            padding: 10px 20px;
                            cursor: pointer;
                            margin-right: 5px;
                        }
                        .tab-button.active {
                            background-color: #ccc;
                        }
                    </style>
                    <script>
                        function showTab(tabId) {
                            var contents = document.getElementsByClassName('tab-content');
                            var buttons = document.getElementsByClassName('tab-button');
                            for (var i = 0; i < contents.length; i++) {
                                contents[i].classList.remove('active');
                                buttons[i].classList.remove('active');
                            }
                            document.getElementById(tabId).classList.add('active');
                            document.querySelector('button[data-tab="'+tabId+'"]').classList.add('active');
                        }
                    </script>
                </head>
                <body>
                    <div class="tabs">
                '''

    # Add buttons for each category
    for i, category in enumerate(categories.keys()):
        active_class = 'active' if i == 0 else ''
        html_code += f'<button class="tab-button {active_class}" onclick="showTab(\'tab-{i}\')" data-tab="tab-{i}">{category}</button>'

    html_code += '</div>'

    # Add content for each category
    for i, (category, embeds) in enumerate(categories.items()):
        active_class = 'active' if i == 0 else ''
        html_code += f'<div id="tab-{i}" class="tab-content {active_class}">'
        for embed in embeds:
            html_code += embed
        html_code += '</div>'

    html_code += '''
                </body>
                </html>
                '''

    return html_code



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

    score = meta['reranking_score']
    if score > 0.8:
        color = "score-green"
    elif score > 0.4:
        color = "score-orange"
    else:
        color = "score-red"

    relevancy_score = f"<p class=relevancy-score>Relevancy score: <span class='{color}'>{score:.1%}</span></p>"

    if meta["chunk_type"] == "text":

        card = f"""
    <div class="card" id="doc{i}">
        <div class="card-content">
            <h2>Doc {i} - {meta['short_name']} - Page {int(meta['page_number'])}</h2>
            <p>{content}</p>
            {relevancy_score}
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
            {relevancy_score}
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



def make_toolbox(tool_name,description = "",checked = False,elem_id = "toggle"):

    if checked:
        span = "<span class='checkmark'>&#10003;</span>"
    else:
        span = "<span class='loader'></span>"

#     toolbox = f"""
# <div class="dropdown">
# <label for="{elem_id}" class="dropdown-toggle">
#     {span}
#     {tool_name}
#     <span class="caret"></span>
# </label>
# <input type="checkbox" id="{elem_id}" hidden/>
# <div class="dropdown-content">
#     <p>{description}</p>
# </div>
# </div>
# """
    

    toolbox = f"""
<div class="dropdown">
<label for="{elem_id}" class="dropdown-toggle">
    {span}
    {tool_name}
</label>
</div>
"""

    return toolbox
