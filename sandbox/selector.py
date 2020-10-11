import math

def splitData(path):
    """ Returns tuple as (testing data, training data)
    """
    with open(path, 'r') as fp:
        lines = fp.readlines()

    split = math.floor(0.8*len(lines))
    return (lines[:split], lines[split:])