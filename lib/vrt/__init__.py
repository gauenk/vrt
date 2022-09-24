from . import original

# -- for loading model --
from .utils.misc import optional
from .augmented import extract_model_io # set input params

def load_model(**kwargs):
    attn_type = optional(kwargs,'attn_type','original')
    if "original" in attn_type:
        return original.load_model(**kwargs)
    else:
        raise ValueError("")

