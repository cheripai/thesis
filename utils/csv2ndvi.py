import pandas as pd
from tkinter import *
from tkinter import filedialog, messagebox
from average_spectra import average_dataframe
from calculate_ndvi import calculateNDVI


class Main(Frame):

    def __init__(self):
        Frame.__init__(self)
        self.master.title("csv2ndvi")
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(5, weight=1)
        self.grid(sticky=W+E+N+S)

        self.df = None
        self.num_spectra = StringVar()
        self.output_filename = StringVar()

        self.num_spectra_label = Label(self, text="Number of Spectra")
        self.num_spectra_label.grid(row=0, column=0, sticky=W)
        self.num_spectra_entry = Entry(self, textvariable=self.num_spectra)
        self.num_spectra_entry.grid(row=1, column=0, sticky=W)

        self.open_file_button = Button(self, text="Browse for CSV file", command=self.open_file)
        self.open_file_button.grid(row=2, column=0, sticky=S)

        self.output_filename_label = Label(self, text="Output file")
        self.output_filename_label.grid(row=0, column=1, sticky=W)
        self.output_filename_entry = Entry(self, textvariable=self.output_filename)
        self.output_filename_entry.grid(row=1, column=1, sticky=W)

        self.calculate_button = Button(self, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=2, column=1, sticky=S)

        self.num_spectra_entry.insert(END, "20")
        self.output_filename_entry.insert(END, "ndvi.txt")

    def open_file(self):
        f = filedialog.askopenfilename()
        if type(f) is str and f != "":
            self.df = pd.read_csv(f, sep=None, engine="python")
        return

    def calculate(self):
        if self.df is None:
            messagebox.showinfo("Error", "Please select a CSV file")
            return
        try:
            num_spectra = int(self.num_spectra.get())
        except:
            messagebox.showinfo("Error", "Invalid value for \"Number of Spectra\"")
            return
        if (self.df.shape[1] - 1) % num_spectra != 0:
            messagebox.showinfo("Error", "Number of spectra not divisble by {}".format(num_spectra))
            return
        averaged = average_dataframe(self.df, num_spectra)
        ndvi = calculateNDVI(averaged)
        ndvi.to_csv(self.output_filename.get(), index=False)
        messagebox.showinfo("File saved!", "File saved to {}".format(self.output_filename.get()))


if __name__ == "__main__":
    Main().mainloop()
