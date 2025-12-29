import numpy as np

def footprint_hist(paths, bbox, nx=120, ny=120):
    """
    bbox = (minlon, minlat, maxlon, maxlat)
    returns grid counts and edges
    """
    minlon, minlat, maxlon, maxlat = bbox

    xs = []
    ys = []
    for p in paths:
        xs.extend(p.lon)
        ys.extend(p.lat)

    xs = np.array(xs)
    ys = np.array(ys)

    # keep inside bbox
    m = (xs>=minlon)&(xs<=maxlon)&(ys>=minlat)&(ys<=maxlat)
    xs = xs[m]; ys = ys[m]

    H, xedges, yedges = np.histogram2d(xs, ys, bins=[nx, ny],
                                      range=[[minlon, maxlon],[minlat, maxlat]])
    # probability
    P = H / (H.sum() + 1e-12)
    return P, xedges, yedges
