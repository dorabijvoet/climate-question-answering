import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

from pyalex import Works, Authors, Sources, Institutions, Concepts, Publishers, Funders
import pyalex

pyalex.config.email = "theo.alvesdacosta@ekimetrics.com"

class OpenAlex():
    def __init__(self):
        pass



    def search(self,keywords,n_results = 100,after = None,before = None):

        if isinstance(keywords,str):
            works = Works().search(keywords)
            if after is not None:
                assert isinstance(after,int), "after must be an integer"
                assert after > 1900, "after must be greater than 1900"
                works = works.filter(publication_year=f">{after}")

            for page in works.paginate(per_page=n_results):
                break

            df_works = pd.DataFrame(page)
            df_works["abstract"] = df_works["abstract_inverted_index"].apply(lambda x: self.get_abstract_from_inverted_index(x))
            df_works["is_oa"] = df_works["open_access"].map(lambda x : x.get("is_oa",False))
            df_works["pdf_url"] = df_works["primary_location"].map(lambda x : x.get("pdf_url",None))
            df_works["content"] = df_works["title"] + "\n" + df_works["abstract"]

        else:
            df_works = []
            for keyword in keywords:
                df_keyword = self.search(keyword,n_results = n_results,after = after,before = before)
                df_works.append(df_keyword)
            df_works = pd.concat(df_works,ignore_index=True,axis = 0)
        return df_works
    

    def rerank(self,query,df,reranker):
    
        scores = reranker.rank(
            query,
            df["content"].tolist(),
            top_k = len(df),
        )
        scores.sort(key = lambda x : x["corpus_id"])
        scores = [x["score"] for x in scores]
        df["rerank_score"] = scores
        return df


    def make_network(self,df):

        # Initialize your graph
        G = nx.DiGraph()

        for i,row in df.iterrows():
            paper = row.to_dict()
            G.add_node(paper['id'], **paper)
            for reference in paper['referenced_works']:
                if reference not in G:
                    pass
                else:
                    # G.add_node(reference, id=reference, title="", reference_works=[], original=False)
                    G.add_edge(paper['id'], reference, relationship="CITING")
        return G

    def show_network(self,G,height = "750px",notebook = True,color_by = "pagerank"):

        net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black",notebook = notebook,directed = True,neighborhood_highlight = True)
        net.force_atlas_2based()

        # Add nodes with size reflecting the PageRank to highlight importance
        pagerank = nx.pagerank(G)

        if color_by == "pagerank":
            color_scores = pagerank
        elif color_by == "rerank_score":
            color_scores = {node: G.nodes[node].get("rerank_score", 0) for node in G.nodes}
        else:
            raise ValueError(f"Unknown color_by value: {color_by}")

        # Normalize PageRank values to [0, 1] for color mapping
        min_score = min(color_scores.values())
        max_score = max(color_scores.values())
        norm_color_scores = {node: (color_scores[node] - min_score) / (max_score - min_score) for node in color_scores}



        for node in G.nodes:
            info = G.nodes[node]
            title = info["title"]
            label = title[:30] + " ..."

            title = [title,f"Year: {info['publication_year']}",f"ID: {info['id']}"]
            title = "\n".join(title)

            color_value = norm_color_scores[node]
            # Generating a color from blue (low) to red (high)
            color = plt.cm.RdBu_r(color_value) # coolwarm is a matplotlib colormap from blue to red
            def clamp(x): 
                return int(max(0, min(x*255, 255)))
            color = tuple([clamp(x) for x in color[:3]])
            color = '#%02x%02x%02x' % color
            
            net.add_node(node, title=title,size = pagerank[node]*1000,label = label,color = color)

        # Add edges
        for edge in G.edges:
            net.add_edge(edge[0], edge[1],arrowStrikethrough=True,color = "gray")

        # Show the network
        if notebook:
            return net.show("network.html")
        else:
            return net


    def get_abstract_from_inverted_index(self,index):

        if index is None:
            return ""
        else:

            # Determine the maximum index to know the length of the reconstructed array
            max_index = max([max(positions) for positions in index.values()])
            
            # Initialize a list with placeholders for all positions
            reconstructed = [''] * (max_index + 1)
            
            # Iterate through the inverted index and place each token at its respective position(s)
            for token, positions in index.items():
                for position in positions:
                    reconstructed[position] = token
            
            # Join the tokens to form the reconstructed sentence(s)
            return ' '.join(reconstructed)