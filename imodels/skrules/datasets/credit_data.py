"""default of credit card clients dataset.

The original database is available from UCI Machine Learning Repository:

    https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

The data contains 30000 observations on 24 variables.

References
----------

Lichman, M. (2013). UCI Machine Learning Repository
[http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer
Science.

"""

import pandas as pd
import numpy as np
from sklearn.datasets.base import get_data_home, Bunch
from sklearn.datasets.base import _fetch_remote, RemoteFileMetadata
from os.path import exists, join


def load_credit_data():
    sk_data_dir = get_data_home()
    archive = RemoteFileMetadata(
        filename='default of credit card clients.xls',
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/'
            '00350/default%20of%20credit%20card%20clients.xls',
        checksum=('30c6be3abd8dcfd3e6096c828bad8c2f'
                  '011238620f5369220bd60cfc82700933'))

    if not exists(join(sk_data_dir, archive.filename)):
        _fetch_remote(archive, dirname=sk_data_dir)

    data = pd.read_excel(join(sk_data_dir, archive.filename),
                         sheet_name='Data', header=1)

    dataset = Bunch(
        data=(data.drop('default payment next month', axis=1)),
        target=np.array(data['default payment next month'])
        )
    return dataset
