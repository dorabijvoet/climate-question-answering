import gradio as gr
import random
from collections import defaultdict

# List of graphs
graphs = [
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/global-warming-by-gas-and-source?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'CO2 & Greenhouse Gas Emissions'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/global-warming-by-gas-and-source?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'Climate Change'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/warming-fossil-fuels-land-use?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'CO2 & Greenhouse Gas Emissions'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/warming-fossil-fuels-land-use?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'Climate Change'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/contributions-global-temp-change?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'CO2 & Greenhouse Gas Emissions'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/contributions-global-temp-change?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'Climate Change'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/carbon-dioxide-emissions-factor?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'CO2 & Greenhouse Gas Emissions'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/carbon-dioxide-emissions-factor?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'Fossil Fuels'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/global-warming-land?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'CO2 & Greenhouse Gas Emissions'}},
    {'embedding': '<iframe src="https://ourworldindata.org/grapher/total-ghg-emissions?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>',
     'metadata': {'source': 'OWID', 'category': 'CO2 & Greenhouse Gas Emissions'}}
]

# Function to randomly select several graphs and organize them by category
def get_graphs_by_category(num_graphs=3):
    selected_graphs = random.sample(graphs, num_graphs)
    graphs_by_category = defaultdict(list)
    
    # Organize graphs by category
    for graph in selected_graphs:
        category = graph['metadata']['category']
        graphs_by_category[category].append(graph['embedding'])
    
    return graphs_by_category

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Random Graph Viewer by Category")
    
    with gr.Tabs():
        with gr.Tab("Current Graphs"):
            checkbox_group = gr.CheckboxGroup(label="Select graphs to save", choices=[])
            button = gr.Button("Show Random Graphs")
            graph_display = gr.HTML()
        
        with gr.Tab("Saved Graphs"):
            saved_graphs_display = gr.HTML()

    saved_graphs = []

    def update_graphs(checked_graphs):
        # Add selected graphs to saved graphs but keep them in current graphs
        global saved_graphs
        saved_graphs.extend([graph for graph in checked_graphs if graph not in saved_graphs])
        
        graphs_by_category = get_graphs_by_category(5)  # Adjust the number as needed
        all_graphs = []
        for category, graphs in graphs_by_category.items():
            category_html = f"<h3 style='color: red; font-weight: bold; font-size: 24px;'>{category}</h3>"
            embeddings = "\n".join(graphs)
            all_graphs.append(category_html + embeddings)
        
        # Combine newly generated graphs and previously kept graphs
        all_graphs_html = "\n".join(all_graphs)
        
        # Update choices for CheckboxGroup
        new_choices = [f"{cat}\n{graph}" for cat, graphs in graphs_by_category.items() for graph in graphs]
        
        # Update saved graphs HTML
        saved_graphs_html = "\n".join(saved_graphs)
        
        return all_graphs_html, gr.CheckboxGroup.update(choices=new_choices), saved_graphs_html
    
    button.click(fn=update_graphs, inputs=checkbox_group, outputs=[graph_display, checkbox_group, saved_graphs_display])

# Launch the app
demo.launch()
