# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:48:12 2020

@author: lcag075
"""

import os
import os.path as op
import xarray as xr
import numpy as np
import gc
from math import gamma as gm
# import wavespectra

import sys


def fix_dir(base_dirs):
    '''
    fix csiro direction for wavespectra (from -> to)'
    '''

    new_dirs = base_dirs + 180
    new_dirs[np.where(new_dirs >= 360)] = new_dirs[np.where(new_dirs >= 360)] - 360

    return new_dirs

def SuperPoint_Superposition(p_stations, stations_id, sectors, deg_sup, st_wind_id):
    '''
    Join station spectral data for each sector

    p_stations   - path to stations database
    stations_id  - list of stations ID
    sectors      - list of tuples: directional sector for each station
    deg_sup      - degrees of superposition
    st_wind_id   - station ID for wind data
    '''

    # generate empty efth_all and cont variables from station dimensions
    st = xr.open_dataset(op.join(p_stations, 'station_{0}.nc'.format(stations_id[0])))
    efth_all = np.full([len(st.time), len(st.frequency), len(st.direction), len(stations_id)], 0.0)
    cont = np.full([len(st.direction)], 0)
    del st

    # TODO sacar concatenado rutas fuera?

    # read stations
    for s_ix, s_id in enumerate(stations_id):
        print('Station: {0}'.format(s_id))

        st = xr.open_dataset(op.join(p_stations, 'station_{0}.nc'.format(s_id)))
        st['direction'] = fix_dir(st['direction'])  # fix direction dimension

        # find station data indexes inside sector (and superposition degrees)
        if (sectors[s_ix][1] - sectors[s_ix][0]) < 0:
            d = np.where((st.direction.values > sectors[s_ix][0] - deg_sup) |
                         (st.direction.values <= sectors[s_ix][1] + deg_sup))[0]
        else:
            d = np.where((st.direction.values > sectors[s_ix][0] - deg_sup) &
                         (st.direction.values <= sectors[s_ix][1] + deg_sup))[0]

        cont[d] += 1
        try:
            efth_all[:,:,d,s_ix] = st.Efth[:,:,d]
        except:
            efth_all[:,:,d,s_ix] = st.efth[:,:,d]

        # get wind data from choosen wind station
        if s_id  == st_wind_id:
            wsp = st.u10m.values
            wdir = st.udir.values
            depth = np.full([len(st.time.values)], st.depth)

    # promediate superimposed station data (using data counter)
    efth_all = (np.sum(efth_all, axis = 3) / cont) * (np.pi / 180)

    # mount superpoint dataset
    super_point = xr.Dataset(
        {
            'efth': (['time','freq','dir'], efth_all),
            'Wspeed': (['time'], wsp),
            'Wdir': (['time'], wdir),
            'Depth': (['time'], depth),
        },
        coords = {
            'time': st.time.values,
            'dir': st.direction.values,
            'freq': st.frequency.values
        }
    )
    # round time to hour
    super_point['time'] = super_point['time'].dt.round('H').values

    return super_point


def BulkParams_Partitions(p_store, sp, chunks=3, wcut=0.333, msw=5, agef=1.7):
    '''
    Calculates superpoint spectra statistics for each partition and bulk parameters using wavespectra library.

    p_store - path for storage
    sp      - superpoint Dataset
    chunks  - split process in N chunks (split by time dimension to prevent memory issues)

    wcut    - wavespectra: wind cut
    msw     - wavespectra: max number of swells
    agef    - wavespectra: age factor
    '''

    # this function needs wavespectra==3.5 

    # get split position
    pos = np.int(len(sp.time) / chunks)

    # solve each chunk
    for p in range(chunks):

        # select current chunk  superpoint data
        if p == 0:
            sp1 = sp.isel(time = np.arange(0, pos))

        elif p == (chunks - 1):
            sp1 = sp.isel(time = np.arange(p*pos, len(sp.time)))

        else:
            sp1 = sp.isel(time = np.arange(p*pos, (p+1)*pos))

        # use wavespectra to calculate spectra partitions
        try:
            ds_part1 = sp1.spec.partition(
                sp1.Wspeed, sp1.Wdir, sp1.Depth,
                wscut = wcut, max_swells = msw, agefac = agef,
            )
        except:
            ds_part1 = sp1.spec.partition(
                sp1.Wspeed, sp1.Wdir, sp1.Depth,
                wscut = wcut, swells = msw, agefac = agef,
            )

        # clean memory
        del sp1
        gc.collect()

        # ensure time dimension is ok
        u, i = np.unique(ds_part1.time, return_index=True)
        ds_part1 = ds_part1.isel(time=i)

        # store solved chunk spectra
        nf = 'partitions_spectra_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut)
        ds_part1.to_netcdf(op.join(p_store, nf))

        # calculate spectral stats
        stats_part1 = ds_part1.spec.stats(['hs','tp','tm02','dpm','dspr'])

        # store spectral stats
        nf = 'partitions_stats_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut)
        stats_part1.to_netcdf(op.join(p_store, nf))

        print('chunk {0}/{1} done.'.format(p+1, chunks))

        # clean memory
        del ds_part1, stats_part1
        gc.collect()


    # load processed chunks spectra stats and merge it
    nfs = ['partitions_stats_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut) for p in range(chunks)]
    stats_part = xr.open_mfdataset([op.join(p_store, f) for f in nfs])

    # calculate superpoint bulk parameters
    bulk_params = sp.spec.stats(['hs','tp','tm02','dpm','dm','dspr'])

    return bulk_params, stats_part



def get_spec_params(part_spec, part_stats):

    dirs = np.arange(5,356,10) # ????

    # get bins from partitioned spectrum
    freqs_rec = part_spec.freq.values
    dirs_rec = part_spec.dir.values
    del part_spec

    # get gamma and directional spreading from spectral wave statistics

    # Seas
    sea = part_stats.sel(part=0)
    a = 1.411
    b = -0.07972
    gamma_sea = np.exp((np.log(sea.tp.values/(a*sea.tm02.values)))/b)
    g_sea = np.nanmean(gamma_sea)
    dspr_sea = np.nanmean(sea.dspr)

    # Swells
    swell = part_stats.sel(part=part_stats.part[1:].values)
    g_swells = np.full([len(dirs)],np.nan)
    dspr_swells = np.full([len(dirs)],np.nan)
    for a in range(len(dirs)):
        s = np.where((swell.dpm.values>(dirs[a]-5)) & (swell.dpm.values<=(dirs[a]+5)))
        dspr_swells[a] = np.nanmean(swell.dspr.values[s[0],s[1]])
        g_swells[a] = np.nanmean(np.exp((np.log(swell.tp.values[s[0],s[1]]/(1.411*swell.tm02.values[s[0],s[1]])))/-0.07972))

    del part_stats

    return dirs, freqs_rec, dirs_rec, g_sea, g_swells, dspr_sea, dspr_swells



def spec_reconstr_snakes(part_spec, part_stats, time_h, hs_swell, tp_swell, dir_swell, sea_params_sim, seaswell='both'):
    '''
    reconstructs spectrum from hourly seas and snakes

    returns spectrum and bulk parameters (wavespectra)
    '''

    # get needed parameters from historical spectrum (partitions and parameters)
    dirs, freqs_rec, dirs_rec, g_sea, g_swells, dspr_sea, dspr_swells = get_spec_params(part_spec, part_stats)

    #if not time_h:
    #    time_h = sea_params_sim.time.values

    # define output
    Sum_Snn = np.zeros((len(dirs_rec),len(freqs_rec),len(time_h)))


    # reconstruct spectrum
    for ind_t, t in enumerate(time_h):

        if seaswell=='snakes':

            hsw = hs_swell[:,ind_t][~np.isnan(hs_swell[:,ind_t])]
            tsw = tp_swell[:,ind_t][~np.isnan(tp_swell[:,ind_t])]
            dsw = dir_swell[:,ind_t][~np.isnan(dir_swell[:,ind_t])]

            hh = hsw
            tt = tsw
            dd = dsw

        if seaswell=='sea':
            hh = np.append(sea_params_sim.Hs_sea.sel(time=t).values, [])
            tt = np.append(sea_params_sim.Tp_sea.sel(time=t).values, [])
            dd = np.append(sea_params_sim.Dir_sea.sel(time=t).values, [])

        if seaswell=='both':

            hsw = hs_swell[:,ind_t][~np.isnan(hs_swell[:,ind_t])]
            tsw = tp_swell[:,ind_t][~np.isnan(tp_swell[:,ind_t])]
            dsw = dir_swell[:,ind_t][~np.isnan(dir_swell[:,ind_t])]

            hh = np.append(sea_params_sim.Hs_sea.sel(time=t).values, hsw)
            tt = np.append(sea_params_sim.Tp_sea.sel(time=t).values, tsw)
            dd = np.append(sea_params_sim.Dir_sea.sel(time=t).values, dsw)


        gamma = np.full([len(hh),1],np.nan)
        dspr = np.full([len(hh),1],np.nan)
        for m in range(len(hh)):
            if m==0 and type!='snakes':
                gamma[m]=g_sea
                dspr[m]=np.deg2rad(dspr_sea)
            else:
                pos=np.nanargmin(np.abs(np.deg2rad(dd[m])-np.deg2rad(dirs)))
                gamma[m]=g_swells[pos] #Default:10/20
                if np.deg2rad(dspr_swells[pos])<0.153:
                    dspr[m]=0.153
                else:
                    dspr[m]=np.deg2rad(dspr_swells[pos])

        s = (2/dspr**2)-1

        for p in range(len(hh)):
            tp = tt[p]
            sigma=np.full([len(freqs_rec),1],0.07)
            sigma[np.where(freqs_rec>(1/tp))]=0.09
            Beta = (0.06238/(0.23+0.0336*gamma[p]-0.185*(1.9+gamma[p])**-1))*(1.094-0.01915*np.log(gamma[p]))

            S = Beta * (hh[p]**2) * (tp**-4) * (freqs_rec**-5)*np.exp(-1.25*(tp*freqs_rec)**-4)*gamma[p]**(np.exp((-(tp*freqs_rec-1)**2)/(2*sigma.T**2)))
            D = ((2**(2*s[p]-1))/np.pi)*(gm(s[p]+1)**2/gm(2*s[p]+1))*np.abs(np.cos((np.deg2rad(dirs_rec)-np.deg2rad(dd[p]))/2))**(2*s[p])
            Snn = np.multiply(S.T,D) * (np.pi/180)
            Snn[np.isnan(Snn)] = 0
            Sum_Snn[:,:,ind_t] = Sum_Snn[:,:,ind_t] + Snn.T


    # to xarray
    spectrum = xr.Dataset({'efth': (['dir','freq','time'],Sum_Snn)},
                          coords={'dir': dirs_rec, 'freq': freqs_rec, 'time':time_h})


    # obtain bulk parameters
    stats = spectrum.spec.stats(['hs','tp','tm02','dpm','dspr'])

    return spectrum, stats



def spec_reconstr_partitions(part_spec, part_stats):


    # get needed parameters from historical spectrum (partitions and parameters)
    dirs, freqs_rec, dirs_rec, g_sea, g_swells, dspr_sea, dspr_swells = get_spec_params(part_spec, part_stats)

    # define output
    Sum_Snn = np.zeros((len(dirs_rec),len(freqs_rec),len(part_stats.time)))

    for rr in range(len(part_stats.time)):
        hh=part_stats.hs.values[~np.isnan(part_stats.tp.values[:,rr]),rr];
        tt=part_stats.tp.values[~np.isnan(part_stats.tp.values[:,rr]),rr];
        dd=part_stats.dpm.values[~np.isnan(part_stats.tp.values[:,rr]),rr];

        if len(hh)>0:

            gamma=np.full([len(hh)],np.nan)
            dspr=np.full([len(hh)],np.nan)
            for m in range(len(hh)):
                if m==0:
                    gamma[m]=g_sea
                    dspr[m]=np.deg2rad(dspr_sea)
                else:
                    pos=np.nanargmin(np.abs(np.deg2rad(dd[m])-np.deg2rad(dirs)))
                    gamma[m]=g_swells[pos] #Default:10/20
                    if np.deg2rad(dspr_swells[pos])<0.153:
                        dspr[m]=0.153
                    else:
                        dspr[m]=np.deg2rad(dspr_swells[pos])

            s = (2/(dspr**2))-1

            for p in range(len(hh)):
                sigma=np.full([len(freqs_rec),1],0.07)
                sigma[np.where(freqs_rec>(1/tt[p]))]=0.09
                Beta = (0.06238/(0.23+0.0336*gamma[p]-0.185*(1.9+gamma[p])**-1))*(1.094-0.01915*np.log(gamma[p]))

                S = Beta * (hh[p]**2) * (tt[p]**-4) * (freqs_rec**-5)*np.exp(-1.25*(tt[p]*freqs_rec)**-4)*gamma[p]**(np.exp((-(tt[p]*freqs_rec-1)**2)/(2*sigma.T**2)))
                D = ((2**(2*s[p]-1))/np.pi)*(gm(s[p]+1)**2/gm(2*s[p]+1))*np.abs(np.cos((np.deg2rad(dirs_rec)-np.deg2rad(dd[p]))/2))**(2*s[p])
                Snn = np.multiply(S.T,D) * (np.pi/180)
                Snn[np.isnan(Snn)] = 0
                Sum_Snn[:,:,rr] = Sum_Snn[:,:,rr] + Snn.T


    # to xarray
    spectrum = xr.Dataset({'efth': (['dir','freq','time'],Sum_Snn)}, coords={'dir': dirs_rec, 'freq': freqs_rec, 'time':part_stats.time})

    # obtain bulk parameters
    stats = spectrum.spec.stats(['hs','tp','tm02','dpm','dspr'])

    return spectrum, stats
