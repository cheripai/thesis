import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.5, style="whitegrid", color_codes=True)
plt.rcParams["figure.figsize"] = (12, 9)

def save_error_chart(y, yhat, value_name, save_path):
    fig = plt.figure()
    plt.plot(range(2), range(2), color="#99CC99", linewidth=2)
    ax = sns.regplot(y, yhat, fit_reg=False)
    plt.title("Measured {} vs. Predicted {}".format(value_name, value_name), y=1.02)
    plt.xlabel("Measured {}".format(value_name))
    plt.ylabel("Predicted {}".format(value_name))
    plt.savefig(save_path)


def normalize(xs):
    """ Puts target data in 0 to 1 range.
    """
    min_x = min(xs)
    max_x = max(xs)
    return [(x - min_x) / (max_x - min_x) for x in xs]

