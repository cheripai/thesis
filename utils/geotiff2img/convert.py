import georasters as gr
import numpy as np
import pandas as pd
import sys
from scipy.misc import imsave

infile = sys.argv[1]

if __name__ == "__main__":
    data = gr.from_file(infile)
    df = data.to_pandas()

    cuts = pd.cut(df["value"], 256)
    df["binned_value"] = [(v.left + v.right) / 2 for v in cuts]

    bin2value = np.sort(df["binned_value"].unique())
    value2bin = {v:b for b, v in enumerate(bin2value)}
    df["bin"] = [value2bin[v] for v in df["binned_value"]]

    img_array = np.zeros((df["row"].max()+1, df["col"].max()+1))
    for _, row in df.iterrows():
        r, c = int(row["row"]), int(row["col"])
        img_array[r, c] = row["bin"]

    imsave("map.png", img_array)
    np.savetxt("bins.txt", bin2value)
