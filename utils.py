import numpy as np
import random
import string
import uuid


def create_user_id():
    """Create user_id
        str: String to id user
    """
    user_id = str(uuid.uuid4())
    return user_id