from sklearn.preprocessing import FunctionTransformer
from collections import defaultdict
from scipy import stats
import numpy as np
from countryinfo import CountryInfo

def get_region(country):
    return CountryInfo(country.replace("_", " ")).region()

def highest_correlation_lag(a, b, bounds=[-20, 20], corrfunc=stats.spearmanr):
    """Lag between two series that yield highest correlation.

    Input
    -----
        a, b : list
            `len(a) == len(b)`.
        bounds : list
            Lag bounds to explore. `len(bounds) == 2`.
        corrfunc : function
            Correlation function to apply
    
    Output
    ------
        out : tuple
            (lag, correlation) tuple
    
    Example:
        >>> highest_correlation_lag(
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 0])
            )
            (-1, 1.0)
    """
    bounds[0] = max(-len(b), bounds[0])
    bounds[1] = min(len(b), bounds[1])
    rvals = []
    for d in range(bounds[0]+2, bounds[1]-1):
        if d < 0:
            a_, b_ = a[:d], b[-d:]
        elif d == 0:
            a_, b_ = a, b
        else:
            a_, b_ = a[d:], b[:-d]
        r = corrfunc(a_, b_)[0]
        r = r if not np.isnan(r) else 0
        rvals.append((d, r))
    return max(rvals, key=lambda kv: kv[1])


def peel_to_confidence(points, confidence=0.8, centroid_func=np.median, scaling_func=None):
    """Peel points in point cloud to get a fraction of the most central points.
    """
    points = points.copy()
    if scaling_func is not None:
        scaler = FunctionTransformer(*scaling_func)
        points = scaler.transform(points)
    N = len(points)
    points = points[~np.any(np.isnan(points), axis=1)]
    while len(points) / N > confidence:
        cent = centroid_func(points)
        dist = np.sqrt(np.sum((points - cent)**2, 1))
        points = np.delete(points, np.argmax(dist), axis=0)
    if scaling_func is not None:
        points = scaler.inverse_transform(points)
    return points


def unzip(zipped_list):
    return list(map(list, zip(*zipped_list)))
