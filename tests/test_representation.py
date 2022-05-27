import os
import random

import numpy as np

class TestTreeTransformer:

    def setup(self):
        '''Test on synthetic dataset
        '''
        np.random.seed(13)
        random.seed(13)
        self.n = 100
        self.p = 2
        self.X = np.random.randn(self.n, self.p)
        self.y = 