import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("filename", metavar="<file>", type=str, help="CSV file to process")
parser.add_argument("-n", type=int, default=40, help="Number of spectra to average")
args = parser.parse_args()

N = args.n
df = pd.read_csv(args.filename)

# Separate label column from data
label = df.iloc[:,0]
df = df.iloc[:,1:]

if (df.shape[1]) % N != 0:
    raise Exception("Number of columns: {}. Not divisible by {}.".format(df.shape[1], N))

# Average every N columns
average_df = pd.DataFrame()
average_df["Wavelengths"] = label
for i in range((df.shape[1]) // N):
    average_df[i] = df.iloc[:,i*N:(i+1)*N].mean(axis=1)

out_filename = args.filename.split(".")[0] + "_avg.csv"
print("Saving to {}".format(out_filename))
average_df.to_csv(out_filename, index=False)
