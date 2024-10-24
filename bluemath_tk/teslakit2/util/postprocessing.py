#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import pandas as pd
import numpy as np

#from ..wts.mjo import categories
from .time_operations import fast_reindex_hourly, fast_reindex_hourly_nsim, \
    xds_further_dates, repair_times_hourly, xds_limit_dates


# TODO: quitar categories de aqui, por que no funciona import???
def categories(rmm1, rmm2, phase):
    '''
    Divides Madden-Julian Oscillation (MJO) data in 25 categories.

    rmm1, rmm2, phase - MJO parameters

    returns array with categories time series
    and corresponding rmm
    '''

    rmm = np.sqrt(rmm1**2 + rmm2**2)
    categ = np.empty(rmm.shape) * np.nan

    for i in range(1,9):
        s = np.squeeze(np.where(phase == i))
        rmm_p = rmm[s]

        # categories
        categ_p = np.empty(rmm_p.shape) * np.nan
        categ_p[rmm_p <=1] =  25
        categ_p[rmm_p > 1] =  i + 8*2
        categ_p[rmm_p > 1.5] =  i + 8
        categ_p[rmm_p > 2.5] =  i
        categ[s] = categ_p

    # get rmm_categ
    rmm_categ = {}
    for i in range(1,26):
        s = np.squeeze(np.where(categ == i))
        rmm_categ['cat_{0}'.format(i)] = np.column_stack((rmm1[s],rmm2[s]))

    return categ.astype(int), rmm_categ

def Generate_HIST_Covariates(AWT,MSL,MJO,DWT,ATD_h, IB):
    '''
    Load, fix, and resample (hourly) all historical covariates:
        AWTs, DWTs, MJO, MMSL, AT
    '''

    # fix WTs id format
    AWT = xr.Dataset({'bmus': AWT.bmus + 1}, coords = {'time': AWT.time})
    DWT = xr.Dataset({'bmus': (('time',), DWT.sorted_bmus_storms + 1)},
                     coords = {'time': DWT.time.values[:]})

    # get MJO categories
    mjo_cs, _ = categories(MJO['rmm1'], MJO['rmm2'], MJO['phase'])
    MJO['bmus'] = (('time',), mjo_cs)

    # reindex data to hourly (pad)
    AWT_h = fast_reindex_hourly(AWT)
    MSL_h = MSL.resample(time='1h').pad()
    MJO_h = fast_reindex_hourly(MJO)
    DWT_h = fast_reindex_hourly(DWT)
    IB_h = IB.resample(time='1h').pad()

    # generate time envelope for output
    d1, d2 = xds_further_dates(
        [AWT_h, ATD_h, MSL_h, MJO_h, DWT_h, ATD_h, IB_h]
    )
    ten = pd.date_range(d1, d2, freq='H')

    # generate empty output dataset
    OUT_h = xr.Dataset(coords={'time': ten})

    # prepare data
    AWT_h = AWT_h.rename({'bmus':'AWT'})
    MJO_h = MJO_h.drop_vars(['mjo','rmm1','rmm2','phase']).rename({'bmus':'MJO'})
    MSL_h = MSL_h.drop_vars(['data_median']).rename({'data_mean':'MMSL'})
    MSL_h['MMSL'] = MSL_h['MMSL'] / 1000.0  # mm to m
    DWT_h = DWT_h.rename({'bmus':'DWT'})
    ATD_h = ATD_h.drop_vars(['WaterLevels','Residual']).rename({'Predicted': 'AT'})
    IB_h = IB_h.drop_vars(['DWT']).rename({'level_IB':'SS'})

    # combine data
    xds = xr.combine_by_coords(
        [OUT_h, AWT_h, MJO_h, MSL_h, DWT_h, ATD_h, IB_h],
        fill_value = np.nan,
    )

    # repair times: round to hour and remove duplicates (if any)
    xds = repair_times_hourly(xds)

    return xds


def Generate_SIM_Covariates(AWT, MSL, MJO, DWT, ATD_h, IB, total_sims=None):

    # optional select total sims
    if total_sims != None:
        AWT = AWT.isel(n_sim=slice(0, total_sims))
        MSL = MSL.isel(n_sim=slice(0, total_sims))
        MJO = MJO.isel(n_sim=slice(0, total_sims))
        DWT = DWT.isel(n_sim=slice(0, total_sims))
        IB = IB.isel(n_sim=slice(0, total_sims))

    # reindex data to hourly (pad)
    AWT_h = fast_reindex_hourly_nsim(AWT)
    MSL_h = fast_reindex_hourly_nsim(MSL)
    MJO_h = fast_reindex_hourly_nsim(MJO)
    DWT_h = fast_reindex_hourly_nsim(DWT)
    IB_h = fast_reindex_hourly_nsim(IB)

    # common dates limits
    d1, d2 = xds_limit_dates([AWT_h, MSL_h, MJO_h, DWT_h, ATD_h, IB_h])
    AWT_h = AWT_h.sel(time = slice(d1, d2))
    MSL_h = MSL_h.sel(time = slice(d1, d2))
    MJO_h = MJO_h.sel(time = slice(d1, d2))
    DWT_h = DWT_h.sel(time = slice(d1, d2))
    ATD_h = ATD_h.sel(time = slice(d1, d2))
    IB_h = IB_h.sel(time = slice(d1, d2))

    # copy to new dataset
    times = AWT_h.time.values[:]
    xds = xr.Dataset(
        {
            'AWT': (('n_sim','time'), AWT_h.evbmus_sims.values[:].astype(int)),
            'MJO': (('n_sim','time'), MJO_h.evbmus_sims.values[:].astype(int)),
            'DWT': (('n_sim','time'), DWT_h.evbmus_sims.values[:].astype(int)),
            'MMSL': (('n_sim','time'), MSL_h.mmsl.values[:]),
            'AT': (('time',), ATD_h.astro.values[:]),
            'SS': (('n_sim','time',), IB_h.level_IB.values[:]),
        },
        coords = {'time': times}
    )

    return xds
