import argparse
import numpy as np
import pandas as pd


RED_RANGE = [655, 700]
NIR_RANGE = [750, 1076]
GREEN1 = 550
YELLOW1 = 680
RED1 = 680
RED3 = 670
NIR1 = 800
NIR3 = 900
NIR4 = 970
NIR5 = 700


def average_ranges(df, red_range, nir_range):
    """ Calculates NDVI using average of reflectance in ranges
    """
    red = df.iloc[red_range[0]:red_range[1],1:].mean(axis=0)
    nir = df.iloc[nir_range[0]:nir_range[1],1:].mean(axis=0)
    return red, nir


def integrate_ranges(df, red_range, nir_range):
    """ Calculates NDVI using integral of reflectance in ranges.
        Does not seem to be very accurate
    """
    red = df.iloc[red_range[0]:red_range[1],1:]
    nir = df.iloc[nir_range[0]:nir_range[1],1:]
    red_area = np.trapz(red, axis=0)
    nir_area = np.trapz(nir, axis=0)
    return pd.Series(red_area), pd.Series(nir_area)


def calculateNDVI(df):
    offset = int(df.iloc[0,0])
    red_range_offset = [r - offset for r in RED_RANGE]
    nir_range_offset = [n - offset for n in NIR_RANGE]

    RED, NIR = average_ranges(df, red_range_offset, nir_range_offset)

    NDVI = (NIR - RED) / (NIR + RED)
    return NDVI


def calculateWBI(df):
    offset = int(df.iloc[0,0])
    nir3 = df.iloc[NIR3-offset,1:]
    nir4 = df.iloc[NIR4-offset,1:]
    return nir4 / nir3


def calculateRedEdge(df):
    offset = int(df.iloc[0,0])
    red3 = df.iloc[RED3-offset,1:]
    nir5 = df.iloc[NIR5-offset,1:]
    return nir5 / red3


def calculateGNDVI(df):
    offset = int(df.iloc[0,0])
    nir1 = df.iloc[NIR1-offset,1:]
    green1 = df.iloc[GREEN1-offset,1:]
    return (nir1 - green1) / (nir1 + green1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", metavar="<file>", type=str, help="CSV file to process")
    args = parser.parse_args()

    df = pd.read_csv(args.filename)
    NDVI = calculateNDVI(df)

    out_filename = args.filename.split(".")[0] + "_ndvi.csv"
    print("Saving to {}".format(out_filename))
    NDVI.to_csv(out_filename, index=False)
