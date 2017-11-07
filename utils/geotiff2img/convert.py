import georasters as gr
import numpy as np
import os
import pandas as pd
import sys
from scipy.misc import imsave

from time import time


def strip_filename(fname):
    return os.path.splitext(os.path.basename(fname))[0]

def geotiff2df(fname):
    try:
        data = gr.from_file(fname)
    except:
        raise Exception("Invalid file!")
    return data.to_pandas()


if __name__ == "__main__":
    infile = sys.argv[1]
    df = geotiff2df(infile)

    cuts = pd.cut(df["value"], 256)
    df["binned_value"] = [(v.left + v.right) / 2 for v in cuts]

    bin2value = np.sort(df["binned_value"].unique())
    value2bin = {v: b for b, v in enumerate(bin2value)}

    img_array = np.zeros((df["row"].max() + 1, df["col"].max() + 1))
    for _, row in df.iterrows():
        r, c = int(row["row"]), int(row["col"])
        img_array[r, c] = value2bin[row["binned_value"]]

    basename = strip_filename(infile)
    imsave(basename + ".png", img_array)
    np.savetxt(basename + "_bins.txt", bin2value, fmt="%1.4f")
