
"""

Inspect Network

"""


# -- sys --
import os
import numpy as np
import pandas as pd
from pathlib import Path

# -- lib --
import vrt
import data_hub

# -- testing --
# import torchvision.models as models
# from dev_basics.trte import test
# import torchinfo
from torchinfo import summary

# -- caching results --
import cache_io

def run_exp(cfg):


    # -- network --
    model = vrt.load_model(cfg)

    # -- data --
    imax = 255.
    data,loaders = data_hub.sets.load(cfg)
    noisy,clean = data_hub.get_sample_pair(data[cfg.dset],cfg.vid_name,cfg.nframes)

    # -- fwd --
    deno = model(noisy)

    # -- view --
    print(model)
    for name,mod in model.named_children():
        print(name)
    batch_size = 1
    summary(model,input_size=(batch_size, 2, 4, 64, 64))

    return {"psnrs":[0.]}

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- configs --
    exps_cfg = {"group0":
                {"dname":["set8"],"dset":["te"],
                 "vid_name":[["tractor"]]},
                "group1":{"warp_mode":["default"]},
                "cfg":
                {"sigma":50,
                 "seed":0,
                 "python_module":"vrt",
                 "isize":"64_64",
                 "task":"rgb_denoise",
                 "nframes":5,
                 "frame_start":0,
                 "frame_end":4,
                 # "depths":[2,]*11,
                 "noise_version":"rgb_noise",
                 "spatial_chunk_size":0,
                 "spatial_chunk_overlap":0.0,
                 "temporal_chunk_size":0,
                 "temporal_chunk_overlap":0,
                 "pretrained_load":False,
                 "append_noise_map":True,
                 "save_deno":False}
                }
    cache_name = ".cache_io/check_vrt"
    records_fn = ".cache_io_pkl/check_vrt.pkl"
    exps = cache_io.exps.unpack(exps_cfg)
    exps,uuids = cache_io.get_uuids(exps,cache_name)
    clear_fxn = lambda x,y: False
    results = cache_io.run_exps(exps,run_exp,uuids=uuids,
                                name=cache_name,enable_dispatch="slurm",
                                records_fn=records_fn,skip_loop=False,
                                records_reload=True,clear=True,
                                clear_fxn=clear_fxn)

    # -- aggregate --
    # results['psnrs'] = results['psnrs'].apply(np.mean)
    # abbrv_path = lambda x: Path(x).stem[-5:]
    # results['pretrained_path'] = results['pretrained_path'].apply(abbrv_path)
    # results = results.sort_values("vid_name")
    # print(results[['psnrs','warp_mode','pretrained_path','vid_name']])
    print("done.")


if __name__ == "__main__":
    main()
