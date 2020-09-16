'''
# Discretization MDLP
Python implementation of Fayyad and Irani's MDLP criterion discretiation algorithm

**Reference:**

Irani, Keki B. "Multi-interval discretization of continuous-valued attributes for classiﬁcation learning." (1993).

**Instructions:**

1. Download Entropy.py and MDLPC.py
2. In a terminal, cd into the directory where the .py files were saved
3. run the following command:
  python MDLPC.py --options=...
  
script options:
* in_path (required): Path to dataset in .csv format (must include header)
* out_path (required): Path where the discretized dataset will be saved
* features (optional): comma-separated list of attribute names to be discretized, e.g., features=attr1,attr2,attr3
* class_label (required): label of class column in .csv dataset
* return_bins (optional): Doesn't take on values. If specified (--return_bins), a text file will be saved in the same directory as out_path. This file will include the description of the bins computed by the algorighm.

**Dependencies:**

1. Pandas
2. Numpy
'''