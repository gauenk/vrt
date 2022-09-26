
# -- api --
from . import original
from . import augmented

# -- for loading model --
from .utils.misc import optional
from .augmented import extract_model_io # set input params

def load_model(**kwargs):
    attn_mode = optional(kwargs,'attn_mode','original')
    if "original" in attn_mode:
        return original.load_model(**kwargs)
    else:
        return augmented.load_model(**kwargs)

