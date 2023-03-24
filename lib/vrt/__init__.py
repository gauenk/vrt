
# -- api --
from . import original
from . import augmented

# -- for loading model --
from .utils.misc import optional
# from .augmented import extract_model_io # set input params
from .augmented import extract_config # set input params
from .augmented import extract_config as extract_model_config # set input params

def load_model(cfg):
    attn_mode = optional(cfg,'attn_mode','augmented')
    if "original" in attn_mode:
        return original.load_model(cfg)
    else:
        return augmented.load_model(cfg)

#
# MISC
#

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    import math
    import numpy as np
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


