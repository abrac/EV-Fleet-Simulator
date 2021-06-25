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

parser = argparse.ArgumentParser()
parser.add_argument('fig_pickle_file')
args = parser.parse_args()

figx = pickle.load(open(args.fig_pickle_file, 'rb'))
figx.show()  # Show the figure, edit it, etc.!
input('Press enter to quit... ')
