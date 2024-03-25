import pandas as pd

from pyalex import Works, Authors, Sources, Institutions, Concepts, Publishers, Funders
import pyalex

pyalex.config.email = "theo.alvesdacosta@ekimetrics.com"

class OpenAlex():
    def __init__(self):
        pass



    def search(self,keywords,n_results = 100,after = None,before = None):
        works = Works().search(keywords).get()

        for page in works.paginate(per_page=n_results):
            break

        df_works = pd.DataFrame(page)

        return works
    

    def make_network(self):
        pass


    def get_abstract_from_inverted_index(self,index):

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