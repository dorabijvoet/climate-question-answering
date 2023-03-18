import numpy as np
import openai
import os
import random
import string


def is_climate_change_related(sentence: str, classifier) -> bool:
    """_summary_

    Args:
        sentence (str): your sentence to classify
        classifier (_type_): zero shot hugging face pipeline classifier

    Returns:
        bool: is_climate_change_related or not
    """
    results = classifier(
        sequences=sentence,
        candidate_labels=["climate change related", "non climate change related"],
    )
    print(f" ## Result from is climate change related {results}")
    return results["labels"][np.argmax(results["scores"])] == "climate change related"


def make_pairs(lst):
    """From a list of even lenght, make tupple pairs
    Args:
        lst (list): a list of even lenght
    Returns:
        list: the list as tupple pairs
    """
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]


def set_openai_api_key(text):
    """Set the api key and return chain.If no api_key, then None is returned.
    To do : add raise error & Warning message
    Args:
        text (str): openai api key
    Returns:
        str: Result of connection
    """
    openai.api_key = os.environ["api_key"]

    if text.startswith("sk-") and len(text) > 10:
        openai.api_key = text
    return f"You're all set: this is your api key: {openai.api_key}"


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    print("Random string of length", length, "is:", result_str)
