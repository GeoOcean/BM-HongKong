#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
from datetime import datetime, timedelta

# pip
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import  genextreme

# teslakit

# import constants
from .config import _faspect, _fsize, _fdpi



# Extremes Return Period for all simulations

def axplot_RP(ax, t_h, v_h, tg_h, vg_h, t_s, v_s, var_name, sim_percentile=95):
    'axes plot return period historical vs simulation'

    # historical maxima
    ax.semilogx(
        t_h, v_h, 'ok',
        markersize = 3, label = 'Historical',
        zorder=9,
    )

    # TODO: fit historical to gev
    # historical GEV fit
    #ax.semilogx(
    #    tg_h, vg_h, '-b',
    #    label = 'Historical - GEV Fit',
    #)

    # simulation maxima - mean
    mn = np.mean(v_s, axis=0)
    ax.semilogx(
        t_s, mn, '-r',
        linewidth = 2, label = 'Simulation (mean)',
        zorder=8,
    )

    # simulation maxima percentiles
    out = 100 - sim_percentile
    p95 = np.percentile(v_s, 100-out/2.0, axis=0,)
    p05 = np.percentile(v_s, out/2.0, axis=0,)

    ax.semilogx(
        t_s, p95, linestyle='-', color='grey',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s, p05, linestyle='-', color='grey',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s, p05, p95, color='lightgray',
        label = 'Simulation ({0}% C.I)'.format(sim_percentile)
    )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('{0}'.format(var_name))
    ax.set_xlim(left=10**0, right=np.max(np.concatenate([t_h,t_s])))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both')

def Plot_ReturnPeriodValidation(xds_hist, xds_sim, sim_percentile=95, show=True):
    'Plot Return Period historical - simulation validation'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # aux func for gev fit
    # TODO: fix it
    def gev_fit(var_fit):
        c = -0.1
        vv = np.linspace(0,10,200)

        sha_g, loc_g, sca_g =  genextreme.fit(var_fit, c)
        pg = genextreme.cdf(vv, sha_g, loc_g, sca_g)

        ix = pg > 0.1
        vv = vv[ix]
        ts = 1/(1 - pg[ix])

        # TODO gev params 95% confidence intervals

        return ts, vv

    # clean nans
    t_r = xds_hist.year.values[:]
    v_r = xds_hist.values[:]

    ix_nan = np.isnan(v_r)
    t_r = t_r[~ix_nan]
    v_r = v_r[~ix_nan]

    # RP calculation, var sorting historical
    t_h = t_rp(t_r)
    v_h = np.sort(v_r)

    # GEV fit historical
    #tg_h, vg_h = gev_fit(v_h)
    tg_h, vg_h = [],[]

    # RP calculation, var sorting simulation
    t_s = t_rp(xds_sim.year.values[:-1])  # remove last year*
    v_s = np.sort(xds_sim.values[:,:-1])  # remove last year*

    # figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    axplot_RP(
        axs,
        t_h, v_h, tg_h, vg_h,
        t_s, v_s,
        xds_sim.name,
        sim_percentile=sim_percentile,
    )

    # show and return figure
    if show: plt.show()
    return fig






# TODO: revisar funciones _cambio climatico


def axplot_RP_CC(ax, t_h, v_h, tg_h, vg_h, t_s, v_s, t_s2, v_s2, var_name, sim_percentile=95,label_1='Simulation', label_2 = 'Simulation Climate Change', ):
    'axes plot return period historical vs simulation'

    # historical maxima
    ax.semilogx(
        t_h, v_h, 'ok',
        markersize = 3, label = 'Historical',
        zorder=9,
    )

    # TODO: fit historical to gev
    # historical GEV fit
    #ax.semilogx(
    #    tg_h, vg_h, '-b',
    #    label = 'Historical - GEV Fit',
    #)

    # simulation maxima - mean
    mn = np.mean(v_s, axis=0)
    ax.semilogx(
        t_s, mn, '-r',
        linewidth = 2, label = '{} (mean)'.format(label_1),
        zorder=8,
    )

    # simulation climate change maxima - mean
    mn2 = np.mean(v_s2, axis=0)
    ax.semilogx(
        t_s2, mn2, '-b',
        linewidth = 2, label = '{} (mean)'.format(label_2),
        zorder=8,
    )

    # simulation maxima percentiles
    out = 100 - sim_percentile
    p95 = np.percentile(v_s, 100-out/2.0, axis=0,)
    p05 = np.percentile(v_s, out/2.0, axis=0,)

    ax.semilogx(
        t_s, p95, linestyle='-', color='grey',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s, p05, linestyle='-', color='grey',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s, p05, p95, color='lightgray',
        #label = 'Simulation ({0}% C.I)'.format(sim_percentile)
        label = '{} ({} C.I)'.format(label_1, sim_percentile)
    )

    # simulation climate change maxima percentiles
    out = 100 - sim_percentile
    p95 = np.percentile(v_s2, 100-out/2.0, axis=0,)
    p05 = np.percentile(v_s2, out/2.0, axis=0,)

    ax.semilogx(
        t_s2, p95, linestyle='-', color='skyblue',
        linewidth = 2, #label = 'Simulation (95% percentile)',
    )

    ax.semilogx(
        t_s2, p05, linestyle='-', color='skyblue',
        linewidth = 2, # label = 'Simulation (05% percentile)',
    )
    ax.fill_between(
        t_s2, p05, p95, color='skyblue', alpha=0.5,
        #label = 'Simulation Climate Change ({0}% C.I)'.format(sim_percentile)
        label = '{} ({} C.I)'.format(label_2, sim_percentile)
    )

    # customize axs
    ax.legend(loc='lower right')
    ax.set_title('Annual Maxima', fontweight='bold')
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel('{0}'.format(var_name))
    ax.set_xlim(left=10**0, right=np.max(np.concatenate([t_h,t_s])))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both')


def Plot_ReturnPeriodValidation_CC(xds_hist, xds_sim, xds_sim2, sim_percentile=95, label_1='Simulation', label_2 = 'Simulation Climate Change', show=True):
    'Plot Return Period historical - simulation validation - simulation CLIMATE CHANGE'

    # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # aux func for gev fit
    # TODO: fix it
    def gev_fit(var_fit):
        c = -0.1
        vv = np.linspace(0,10,200)

        sha_g, loc_g, sca_g = genextreme.fit(var_fit, c)
        pg = genextreme.cdf(vv, sha_g, loc_g, sca_g)

        ix = pg > 0.1
        vv = vv[ix]
        ts = 1/(1 - pg[ix])

        # TODO gev params 95% confidence intervals

        return ts, vv

    # clean nans
    t_r = xds_hist.year.values[:]
    v_r = xds_hist.values[:]

    ix_nan = np.isnan(v_r)
    t_r = t_r[~ix_nan]
    v_r = v_r[~ix_nan]

    # RP calculation, var sorting historical
    t_h = t_rp(t_r)
    v_h = np.sort(v_r)

    # GEV fit historical
    #tg_h, vg_h = gev_fit(v_h)
    tg_h, vg_h = [],[]

    # RP calculation, var sorting simulation
    t_s = t_rp(xds_sim.year.values[:-1])  # remove last year*
    v_s = np.sort(xds_sim.values[:,:-1])  # remove last year*

    t_s2 = t_rp(xds_sim2.year.values[:-1])  # remove last year*
    v_s2 = np.sort(xds_sim2.values[:,:-1])  # remove last year*

    # figure
    fig, axs = plt.subplots(figsize=(_faspect*_fsize, _fsize))

    axplot_RP_CC(
        axs,
        t_h, v_h, tg_h, vg_h,
        t_s, v_s,
        t_s2, v_s2,
        xds_sim.name,
        sim_percentile=sim_percentile,
        label_1 = label_1,
        label_2 = label_2,
    )

    # show and return figure
    if show: plt.show()
    return fig
