

import numpy as np

def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict: return pydict[field]
    else: return default

def select_sigma(data_sigma):
    model_sigma_list = np.array([15,25,50])
    idx = np.argmin(np.abs(model_sigma_list - data_sigma))
    model_sigma = model_sigma_list[idx]
    return model_sigma

def dcat(dict1,dict2):
    if dict2 is None: return
    for key,val in dict2.items():
        dict1[key] = val

