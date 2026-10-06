"""Tidy layout for trees with variable-size nodes (contour-based Reingold-Tilford)."""


def tidy(root, kids, breadth, depth_of, depth_size, gap=18.0, level_gap=60.0):
    """Place nodes so that no two overlap and parents sit centered over children.

    Parameters
    ----------
    root : node id
    kids : dict id -> list of visible child ids
    breadth : dict id -> node extent across the layout direction
    depth_of : dict id -> integer level
    depth_size : dict id -> node extent along the layout direction
    Returns (center positions along breadth, top positions along depth, total breadth, total depth).
    """
    rel = {}

    def place(n):
        cs = kids.get(n) or []
        half = breadth[n] / 2.0
        if not cs:
            return [(-half, half)]
        conts = [place(c) for c in cs]
        offs, acc = [0.0], list(conts[0])
        for c in conts[1:]:
            m = min(len(acc), len(c))
            shift = max(acc[i][1] - c[i][0] for i in range(m)) + gap
            offs.append(shift)
            merged = []
            for i in range(max(len(acc), len(c))):
                a = acc[i] if i < len(acc) else None
                b = (c[i][0] + shift, c[i][1] + shift) if i < len(c) else None
                merged.append((min(a[0], b[0]), max(a[1], b[1])) if a and b else (a or b))
            acc = merged
        mid = (offs[0] + offs[-1]) / 2.0
        for c, o in zip(cs, offs):
            rel[c] = o - mid
        return [(-half, half)] + [(lo - mid, hi - mid) for lo, hi in acc]

    cont = place(root)
    pos, stack = {root: 0.0}, [root]
    while stack:
        n = stack.pop()
        for c in kids.get(n) or []:
            pos[c] = pos[n] + rel[c]
            stack.append(c)
    lo = min(a for a, _ in cont)
    hi = max(b for _, b in cont)
    pos = {k: v - lo for k, v in pos.items()}

    levels = {}
    for n in pos:
        d = depth_of[n]
        levels[d] = max(levels.get(d, 0.0), depth_size[n])
    offs, acc = {}, 0.0
    for d in sorted(levels):
        offs[d] = acc
        acc += levels[d] + level_gap
    top = {n: offs[depth_of[n]] for n in pos}
    return pos, top, hi - lo, acc - level_gap
