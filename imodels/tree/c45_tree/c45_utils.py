import math


def prettify(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            prettify(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem


def is_numeric_feature(attr):
    for x in set(attr):
        if not x == "?":
            try:
                x = float(x)
                return isinstance(x, float)
            except ValueError:
                return False
    return True


def set_as_leaf_node(parent, y_str):
    num_max = 0
    for cat in set(y_str):
        num_cat = y_str.count(cat)
        if num_cat > num_max:
            num_max = num_cat
            most_cat = cat
    parent.text = most_cat


def entropy(x):
    ent = 0
    for k in set(x):
        p_i = float(x.count(k)) / len(x)
        ent = ent - p_i * math.log(p_i, 2)
    return ent


def gain_ratio(category, attr):
    s = 0
    cat = []
    att = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cat.append(category[i])
            att.append(attr[i])
    for i in set(att):
        p_i = float(att.count(i)) / len(att)
        cat_i = []
        for j in range(len(cat)):
            if att[j] == i:
                cat_i.append(cat[j])
        s = s + p_i * entropy(cat_i)
    gain = entropy(cat) - s
    ent_att = entropy(att)
    if ent_att == 0:
        return 0
    else:
        return gain / ent_att


def gain(category, attr):
    cats = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cats.append([float(attr[i]), category[i]])
    cats = sorted(cats, key=lambda x: x[0])

    cat = [cats[i][1] for i in range(len(cats))]
    att = [cats[i][0] for i in range(len(cats))]
    if len(set(att)) == 1:
        return 0
    else:
        gains = []
        div_point = []
        for i in range(1, len(cat)):
            if not att[i] == att[i - 1]:
                gains.append(entropy(cat[:i]) * float(i) / len(cat) + entropy(cat[i:]) * (1 - float(i) / len(cat)))
                div_point.append(i)
        gain = entropy(cat) - min(gains)

        p_1 = float(div_point[gains.index(min(gains))]) / len(cat)
        ent_attr = -p_1 * math.log(p_1, 2) - (1 - p_1) * math.log((1 - p_1), 2)
        return gain / ent_attr


def get_best_split(category, attr):
    cats = []
    for i in range(len(attr)):
        if not attr[i] == "?":
            cats.append([float(attr[i]), category[i]])
    cats = sorted(cats, key=lambda x: x[0])

    cat = [cats[i][1] for i in range(len(cats))]
    att = [cats[i][0] for i in range(len(cats))]
    gains = []
    split_point = []
    for i in range(1, len(cat)):
        if not att[i] == att[i - 1]:
            gains.append(entropy(cat[:i]) * float(i) / len(cat) + entropy(cat[i:]) * (1 - float(i) / len(cat)))
            split_point.append(i)
    return att[split_point[gains.index(min(gains))]]


def add(d1, d2):
    d = d1
    for i in d2:
        if d.has_key(i):
            d[i] = d[i] + d2[i]
        else:
            d[i] = d2[i]
    return d


def decision(root, obs, feature_names: list, p):
    if root.hasChildNodes():
        att_name = root.firstChild.nodeName
        if att_name == "#text":
            return decision(root.firstChild, obs, feature_names, p)
        else:
            att = obs[feature_names.index(att_name)]
            if att == "?":
                d = {}
                for child in root.childNodes:
                    d = add(d, decision(child, obs, feature_names, p * float(child.getAttribute("p"))))
                return d
            else:
                for child in root.childNodes:
                    if child.getAttribute("flag") == "m" and child.getAttribute("feature") == att or \
                            child.getAttribute("flag") == "l" and float(att) < float(child.getAttribute("feature")) or \
                            child.getAttribute("flag") == "r" and float(att) >= float(child.getAttribute("feature")):
                        return decision(child, obs, feature_names, p)
    else:
        return {root.nodeValue: root.parentNode.getAttribute('p')}
