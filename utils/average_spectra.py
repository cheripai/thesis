import argparse
import pandas as pd


def average_dataframe(df, n):
    # Separate label column from data
    label = df.iloc[:,0]
    df = df.iloc[:,1:]

    if (df.shape[1]) % n != 0:
        raise Exception("Number of columns: {}. Not divisible by {}.".format(df.shape[1], n))

    # Average every N columns
    average_df = pd.DataFrame()
    average_df["Wavelengths"] = label
    for i in range((df.shape[1]) // n):
        average_df[i] = df.iloc[:,i*n:(i+1)*n].mean(axis=1)

    return average_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", metavar="<file>", type=str, help="CSV file to process")
    parser.add_argument("-n", type=int, default=20, help="Number of spectra to average")
    args = parser.parse_args()

    N = args.n
    df = pd.read_csv(args.filename)
    average_df = average_dataframe(df, N)

    out_filename = args.filename.split(".")[0] + "_avg.csv"
    print("Saving to {}".format(out_filename))
    average_df.to_csv(out_filename, index=False)
