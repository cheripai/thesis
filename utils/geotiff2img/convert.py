import georasters as gr
import numpy as np
import os
import pandas as pd
import sys
from scipy.misc import imsave


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

    min_value = df["value"].min()
    max_value = df["value"].max()

    bins = np.zeros(256)
    for i in range(256):
        bins[i] = (max_value - min_value) / 256 * i + min_value

    img_array = np.zeros((df["row"].max() + 1, df["col"].max() + 1))
    for _, row in df.iterrows():
        r, c = int(row["row"]), int(row["col"])
        img_array[r, c] = np.argmin(np.abs(bins-row["value"]))

    basename = strip_filename(infile)
    imsave(basename + ".png", img_array)
    np.savetxt(basename + "_bins.txt", bins, fmt="%1.4f")
