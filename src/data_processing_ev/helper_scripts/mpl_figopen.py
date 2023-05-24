#!/usr/bin/env python3
"""
Open Matplotlib `fig.pickle` files.

They can be created by:

```
import pickle
pickle.dump(fig, open(output_filepath, 'wb'))
```
"""

import pickle
import argparse

from matplotlib import pyplot

pyplot.ion()

def mpl_figopen():
    parser = argparse.ArgumentParser()
    parser.add_argument('fig_pickle_file')
    args = parser.parse_args()

    figx = pickle.load(open(args.fig_pickle_file, 'rb'))
    pyplot.show(block=True)


if __name__ == "__main__":
    mpl_figopen()
