"""

Compare the impact of train/test using exact/refineimate methods


"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- testing --
from dev_basics.trte import test

# -- plotting --
# import apsearch
# from apsearch import plots

# -- caching results --
import cache_io

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- configs --
    exps_cfg = {"group0":
                {"dname":["set8"],"dset":["te"],
                 "vid_name":["tractor","hypersmooth"]},
                # {"dname":["davis"],"dset":["val"],
                #  "vid_name":["bike-packing"]},
                "group1":{"warp_mode":["default","dnls"]},
                "cfg":
                {"sigma":30,
                 "seed":0,
                 "python_module":"vrt",
                 "isize":"128_128",
                 "task":"rgb_denoise",
                 "nframes":5,
                 "frame_start":0,
                 "frame_end":4,
                 "noise_version":"rgb_noise",
                 "spatial_chunk_size":0,
                 "spatial_chunk_overlap":0.0,
                 "temporal_chunk_size":0,
                 "temporal_chunk_overlap":0,
                 "pretrained_path":"model_zoo/vrt/008_VRT_videodenoising_DAVIS.pth",
                 "pretrained_load":True,
                 "pretrained_type":"mod",
                 "append_noise_map":True,
                 "save_deno":False}
                }
    cache_name = ".cache_io/test_davis"
    records_fn = ".cache_io_pkl/test_davis.pkl"
    exps = cache_io.exps.unpack(exps_cfg)
    exps,uuids = cache_io.get_uuids(exps,cache_name)
    clear_fxn = lambda x,y: x==1
    results = cache_io.run_exps(exps,test.run,uuids=uuids,
                                name=cache_name,enable_dispatch="slurm",
                                records_fn=records_fn,skip_loop=False,
                                records_reload=True,clear=False,
                                clear_fxn=clear_fxn)

    # -- aggregate --
    results['psnrs'] = results['psnrs'].apply(np.mean)
    print(results[['psnrs','warp_mode']])


if __name__ == "__main__":
    main()
