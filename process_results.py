from itertools import islice, cycle

import matplotlib.pyplot as plt
import numpy as np
from numpy.matlib import rand
from pandas import DataFrame



def create_diagram():

    df2 = DataFrame([[1, 2, 3, 4, 5, 6]], columns=['SVM', 'LSTM', 'TD_LSTM', 'MemNet', 'RAM', 'IAN'])
    my_colors = list(islice(['b', 'red', 'g', 'y', 'k'], None, len(df2)))
    plt.figure(figsize=(3, 8))
    df2.plot(kind='bar', color=my_colors)
    plt.show()
    plt.savefig("output/adr_precision.png")
    print("End")


create_diagram()