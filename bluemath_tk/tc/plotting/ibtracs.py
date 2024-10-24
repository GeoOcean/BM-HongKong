#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO doc

import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# matplotlib

from ..ibtracs import data_to_discret


def Plot_ibtracs_sst_mlp_pmin(df):
    '''
    Plot all the sst-mld-pressure data from the ibtracs dataframe from
    add_sst_mld (it includes information of the id of the storm, longitude, latitude, pressure, time of
    the tc tracks included in IBTrACS; mld and sst ) and the same data but discretized in
    intervals of 0.5ºC and 0.5m for sst and mld respectively'''

    plt.subplot(231)
    plt.scatter(
        df['sst'], df['mld'],
        s = 1,
        c = df['pres'],
        cmap = 'plasma',
    )
    plt.xlabel('SST (ºC)')
    plt.ylabel('MLD (m)')
    plt.colorbar(label = 'Pressure (mbar)')

    plt.subplot(232)
    xx, yy, zz = data_to_discret(
        df['sst'].values, df['mld'].values, 0.5, 5,
        df['pres'].values, 16, 32, 0, 175,
        option = 'min',
    )
    plt.pcolormesh(
        xx, yy, zz,
        cmap = 'plasma',
    )
    plt.xlabel('SST (ºC)')
    plt.ylabel('MLD (m)')
    plt.colorbar(label='Pmin (mbar)')

    # TODO y estos subplots?
    '''
    plt.subplot(234)
    xx,yy,zz = data_to_discret(df['sst'].values, df['mld'].values, 0.5, 5, df['pres'].values, 16, 32, 0, 175, option='std')
    plt.pcolormesh(xx, yy, zz, cmap='Blues'); plt.xlabel('SST (ºC)'); plt.ylabel('MLD (m)'); plt.colorbar(label='Pstd (mbar)')

    plt.subplot(235)
    xx,yy,zz = data_to_discret(df['sst'].values, df['mld'].values, 0.5, 5, df['pres'].values, 16, 32, 0, 175, option='num')
    plt.pcolormesh(xx, yy, zz, cmap='hot_r'); plt.xlabel('SST (ºC)'); plt.ylabel('MLD (m)'); plt.colorbar(label='Number of occurrence')
    '''

    plt.gcf().set_size_inches(20, 8)

def Plot_index(xx, yy, zz, index_zz):
    '''
    plot the resultant tailor-made index
    '''

    fig = plt.figure(figsize = (10,6))

    plt.pcolormesh(xx, yy, index_zz, cmap='plasma')

    plt.xlabel('SST (ºC)')
    plt.ylabel('MLD (m)')
    plt.title('Index limits: min({0}) max({1})'.format(np.nanmin(zz), np.nanmax(zz)))
    plt.colorbar(label = 'Index []')

    return fig

