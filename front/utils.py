
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
