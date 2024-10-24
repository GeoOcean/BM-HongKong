#!/usr/bin/env python
# -*- coding: utf-8 -*-


# pip
import numpy as np


def Persistences(series):
    'Return series persistences for each element'

    # locate dates where series changes
    s_diff = np.diff(series)
    ix_ch = np.where((s_diff != 0))[0]+1
    ix_ch = np.insert(ix_ch, 0, 0)

    wt_ch = series[ix_ch][:-1] # bmus where WT changes
    wt_dr = np.diff(ix_ch)

    # output dict
    d_pers = {}
    for e in set(series):
        d_pers[e] = wt_dr[wt_ch==e]

    return d_pers

def ClusterProbabilities(series, set_values):
    'return series probabilities for each item at set_values'

    us, cs = np.unique(series, return_counts=True)
    d_count = dict(zip(us,cs))

    # cluster probabilities
    cprobs = np.zeros((len(set_values)))
    for i, c in enumerate(set_values):
       cprobs[i] = 1.0*d_count[c]/len(series) if c in d_count.keys() else 0.0

    return cprobs

def ChangeProbabilities(series, set_values):
    'return series transition count and probabilities'

    # count matrix
    count = np.zeros((len(set_values), len(set_values)))
    for ix, c1 in enumerate(set_values):
        for iy, c2 in enumerate(set_values):

            # count cluster-next_cluster ocurrences
            us, cs = np.unique((series[:-1]==c1) & (series[1:]==c2), return_counts=True)
            d_count = dict(zip(us,cs))
            count[ix, iy] = d_count[True] if True in d_count.keys() else 0

    # probabilities
    probs = np.zeros((len(set_values), len(set_values)))
    for ix, _ in enumerate(set_values):

        # calculate each row probability
        probs[ix,:] = count[ix,:] / np.sum(count[ix, :])

    return count, probs
