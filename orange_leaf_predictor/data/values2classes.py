import numpy as np
import sys
from collections import Counter
from random import shuffle


def values2classes(values):
    classes = []
    p1 = np.nanmean(values) - np.nanstd(values)
    p2 = np.nanmean(values) + np.nanstd(values)
    for value in values:
        if value < p1:
            classes.append(0)
        elif value >= p1 and value < p2:
            classes.append(1)
        elif value >= p2:
            classes.append(2)
        else:
            classes.append(np.NAN)
    return classes


def balance_classes(classes, slack_p=0.2):
    counter = Counter()
    for value in classes:
        if value is np.NAN:
            continue
        counter[value] += 1
    counts = list(counter.values())

    min_count = min(counts)
    slack = int(np.round(min_count * slack_p))

    discard = {}
    for class_, count in counter.items():
        if count > min_count + slack:
            discard[class_] = list(range(count))
            shuffle(discard[class_])
            discard[class_] = discard[class_][:count-min_count-slack]

    counter = Counter()
    balanced = []
    for value in classes:
        if value is np.NAN:
            balanced.append(np.NAN)
            continue

        counter[value] += 1

        if value in discard and counter[value] in discard[value]:
            balanced.append(np.NAN)
        else:
            balanced.append(value)

    return balanced 
                



if __name__ == "__main__":
    values = []
    with open(sys.argv[1]) as f:
        for value in f.readlines():
            try:
                values.append(float(value))
            except ValueError:
                values.append(np.NAN)

    values = np.array(values)
    classes = values2classes(values)
    balanced = balance_classes(classes)

    with open(sys.argv[2], "w+") as f:
        for value in balanced:
            f.write(str(value) + "\n")
