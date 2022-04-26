import logging
import time

import numpy as np

from imodels import OptimalTreeClassifier
from imodels.util.data_util import get_clean_dataset

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("GOSDT")
DATASETS_CLASSIFICATION = [
    # classification datasets from original random forests paper
    # page 9: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
    # ("sonar", "sonar", "pmlb"),
    # ("heart", "heart", 'imodels'),
    # ("breast-cancer", "breast_cancer", 'imodels'),
    # ("haberman", "haberman", 'imodels'),
    ("ionosphere", "ionosphere", 'pmlb'),
    ("diabetes", "diabetes", "pmlb"),
    # # #("liver", "8", "openml"), # note: we omit this dataset bc it's label was found to be incorrect (see caveat here: https://archive.ics.uci.edu/ml/datasets/liver+disorders#:~:text=The%207th%20field%20(selector)%20has%20been%20widely%20misinterpreted%20in%20the%20past%20as%20a%20dependent%20variable%20representing%20presence%20or%20absence%20of%20a%20liver%20disorder.)
    # # #("credit-g", "credit_g", 'imodels'), # like german-credit, but more feats
    # ("german-credit", "german", "pmlb"),
    #
    # #clinical-decision rules
    # #("iai-pecarn", "iai_pecarn.csv", "imodels"),
    #
    # #popular classification datasets used in rule-based modeling / fairness
    # # page 7: http://proceedings.mlr.press/v97/wang19a/wang19a.pdf
    # ("juvenile", "juvenile_clean", 'imodels'),
    # ("recidivism", "compas_two_year_clean", 'imodels'),
    # # ("credit", "credit_card_clean", 'imodels'),
    # # ("readmission", 'readmission_clean', 'imodels'),  # v big
]

if __name__ == '__main__':
    for d in DATASETS_CLASSIFICATION:
        for reg in [0.001, 0.0001, 0.00001, 0]:
            s = time.time()
            gosdt_cls = OptimalTreeClassifier(regularization=0.00001)
            X, y, feat_names = get_clean_dataset(d[1], data_source=d[2])
            LOGGER.info(f"Data dim (n, p) = {X.shape}")
            gosdt_cls.fit(X, y)
            LOGGER.info(f"Complexity: {gosdt_cls.complexity_}, Reg: {reg}, Time: {np.round(time.time() - s,2)}")



