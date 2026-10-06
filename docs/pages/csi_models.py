"""Draw the cervical-spine-injury models shown in figs.html (Figs 2 and 3) and shrinkage.html (Fig 2)
with imodels.viz, using the same data, split and settings as each post's quickstart code.

Writes docs/img/figs_csi_model_small.svg, figs_csi_model_large.svg and shrinkage_csi_model.svg:

    uv run python docs/pages/csi_models.py
"""

import os

from sklearn.model_selection import train_test_split

from imodels import FIGSClassifier, HSTreeClassifierCV, get_clean_dataset, viz

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")

X, y, feat_names = get_clean_dataset("csi_pecarn_pred")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
kw = dict(feature_names=feat_names, class_names=["no CSI", "CSI"], simple=True)

for name, model, title in [
    ("figs_csi_model_small", FIGSClassifier(max_rules=4), "Cervical spine injury (FIGS, 4 rules)"),
    ("figs_csi_model_large", FIGSClassifier(), "Cervical spine injury (FIGS, no rule limit)"),
    ("shrinkage_csi_model", HSTreeClassifierCV(max_leaf_nodes=7), "Cervical spine injury (hierarchical shrinkage)"),
]:
    model.fit(X_train, y_train, feature_names=feat_names)
    viz.draw(model, X_train, y_train, title=title, **kw).save(os.path.join(IMG, name + ".svg"))
    print("wrote", name + ".svg")
