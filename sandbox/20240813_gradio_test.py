import gradio as gr

# The list of graphs
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

# Organize graphs by category
categories = {}
for graph in graphs:
    category = graph['metadata']['category']
    if category not in categories:
        categories[category] = []
    categories[category].append(graph['embedding'])

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Tabs():
        for category, embeddings in categories.items():
            with gr.Tab(category):
                with gr.Row():
                    for embedding in embeddings:
                        with gr.Column(scale=1):  # Each graph gets its own column
                            gr.HTML(embedding)

# Launch the interface
demo.launch()
