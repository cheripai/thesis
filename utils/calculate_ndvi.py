import argparse
import pandas as pd


RED_RANGE = [655, 700]
NIR_RANGE = [750, 1076]


parser = argparse.ArgumentParser()
parser.add_argument("filename", metavar="<file>", type=str, help="CSV file to process")
args = parser.parse_args()

df = pd.read_csv(args.filename)

offset = int(df.iloc[0,0])
RED_RANGE = [r - offset for r in RED_RANGE]
NIR_RANGE = [n - offset for n in NIR_RANGE]

RED = df.iloc[RED_RANGE[0]:RED_RANGE[1],1:].mean(axis=0)
NIR = df.iloc[NIR_RANGE[0]:NIR_RANGE[1],1:].mean(axis=0)

NDVI = (NIR - RED) / (NIR + RED)

out_filename = args.filename.split(".")[0] + "_ndvi.csv"
print("Saving to {}".format(out_filename))
NDVI.to_csv(out_filename, index=False)
