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
    button = gr.Button("Show Random Graphs")
    
    # Create tabs for each possible category
    with gr.Tabs() as tabs:
        graph_displays = {}

        for category in set(graph['metadata']['category'] for graph in graphs):
            with gr.Tab(category):
                graph_displays[category] = gr.HTML()

    def update_graphs():
        graphs_by_category = get_graphs_by_category(5)  # Adjust the number as needed
        updates = {}
        for category, graphs in graphs_by_category.items():
            embeddings = "\n".join(graphs)
            updates[graph_displays[category]] = embeddings
        return updates
    
    button.click(fn=update_graphs, outputs=[graph_displays[category] for category in graph_displays])

# Launch the app
demo.launch()
