#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO doc 
# TODO rename ds_*
# TODO completa revision y refactor de este modulo

import os.path as op
from datetime import datetime
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr


# TODO duplicado en seasonal_forecast_tcs.plotting.tcs
def get_category(ycpres):
    'Defines storm category according to minimum pressure centers'

    categ = []
    for i in range(len(ycpres)):
        if (ycpres[i] == 0) or (np.isnan(ycpres[i])):
            categ.append(-1)
        elif ycpres[i] < 920:  categ.append(5)
        elif ycpres[i] < 944:  categ.append(4)
        elif ycpres[i] < 964:  categ.append(3)
        elif ycpres[i] < 979:  categ.append(2)
        elif ycpres[i] < 1000: categ.append(1)
        elif ycpres[i] >= 1000: categ.append(0)

    return categ

def data_to_discret(var_x, var_y, dx, dy, var_color,
                    xmin, xmax, ymin, ymax, option='mean'):
    '''
    Data is discretized into clases for pcolormesh plot
    '''

    # axis discretization
    X = np.arange(xmin, xmax + dx, dx)
    Y = np.arange(ymin, ymax + dy, dy)

    # meshgrid
    XX, YY = np.meshgrid(X, Y)

    var_discret = np.nan * np.ones(np.shape(XX))

    for j,xj in enumerate(XX[0][:-1]):
        for i,yi in enumerate(YY[:,0][:-1]):

            pos = np.where((var_x>=xj) & (var_x<=XX[0][j+1]) & (var_y>=yi) & (var_y<=YY[:,0][i+1]))
            if pos[0].size > 0:
                if option=='mean':
                    var_discret[i,j] = np.nanmean(var_color[pos])
                elif option == 'min':
                    var_discret[i,j] = np.nanmin(var_color[pos])
                elif option == 'std':
                    var_discret[i,j] = np.nanstd(var_color[pos])
                elif option == 'num':
                    var_discret[i,j] = len(np.where(~np.isnan(var_color[pos]))[0])
#                elif option == 'adjust':
#                    #Linear model regression  Y = alpha * X
#                    x0 = np.ones(1)
#                    res_lsq = least_squares(modelfun, x0,
#                        args = (var_x[pos], var_y[pos]))

#                    # check model at fitting period
#                    y_0s = np.zeros(var_y[pos].shape)
#                    y_p = modelfun(
#                            res_lsq.x, var_x[pos], y_0s)

#                    var_discret[i,j] =  res_lsq

            else: continue

    return XX, YY, var_discret

def load_storms_sp(p_ibtracs):
    '''
    Obtain the storms of the South Pacific and the South Indian basins
    from IBTrACs database

    p_ibtracks - path to ibtracs database file
    '''

    # IBTrACS v4.0
    xds_ibtracs = xr.open_dataset(p_ibtracs)
    basin = xds_ibtracs.basin.values

    # basin IDs (b'EP' b'NA' b'NI' b'SA' b'SI' b'SP' b'WP')
    storms_SP = np.unique(np.where(basin == np.bytes_(b'SP'))[0])
    storms_SI = np.unique(np.where(basin == np.bytes_(b'SI'))[0])

    # remove intersect positions (SI basin)
    _, common_ind, _ = np.intersect1d(storms_SP, storms_SI, return_indices=True)
    storms_SP = np.delete(storms_SP, common_ind)

    # SP subset
    xds_SP = xds_ibtracs.sel(storm=storms_SP)

    print('All basins storms: ', xds_ibtracs.storm.shape[0])
    print('SP basin storms:   ', xds_SP.storm.shape[0], '\n')

    return xds_ibtracs, xds_SP

def genesis_cat_sp(xds_SP, xds_kma):
    '''
    TODO doc
    '''

    # extract South Pacific storms genesis & category
    st_lons = xds_SP.lon.values
    st_lons[st_lons<0] += 360     # longitude convention
    st_lats = xds_SP.lat.values
    st_pres = xds_SP.wmo_pres.values
    st_times = xds_SP.time.values.astype('datetime64[D]')

    st_categ = np.zeros((st_pres.shape))*np.nan
    for i in range(st_pres.shape[0]):    st_categ[i,:] = get_category(st_pres[i,:])

    # assign storms to each DWTs 
    st_bmus = np.zeros((st_times.shape))*np.nan
    bmus_time = xds_kma.time.values.astype('datetime64[D]')

    for i in range(st_bmus.shape[0]):
        for j in range(st_bmus.shape[1]):
            st_ij = st_times[i,j]
            it = np.where(bmus_time==st_ij)[0][:]
            if it.size > 0:
                st_bmus[i,j] = xds_kma.bmus.values[it]

    return st_bmus

def extract_tcs_rectangle(xds_TCs, lon1, lon2, lat1, lat2, d_vns):
    '''
    Extracts TCs inside a rectangle - used with NWO or Nakajo databases

    xds_TCs: tropical cyclones track database
        lon, lat, pressure variables
        storm dimension

    rectangle defined by:
        lon1, lon2  -  longitudes
        lat1, lat2  -  latitudes

    d_vns: dictionary to set longitude, latitude, time and pressure varnames

    returns:
        xds_area: selection of xds_TCs inside circle
        xds_inside: contains TCs custom data inside circle
    '''

    # get names of vars: longitude, latitude, pressure and time
    nm_lon = d_vns['longitude']
    nm_lat = d_vns['latitude']

    # storms longitude, latitude, pressure and time (if available)
    lon = xds_TCs[nm_lon].values[:]
    lat = xds_TCs[nm_lat].values[:]

    # get storms inside circle area
    n_storms = xds_TCs.storm.shape[0]
    l_storms_area = []

    for i_storm in range(n_storms):

        # fix longitude <0 data and skip "one point" tracks
        lon_storm = lon[i_storm]
        if not isinstance(lon_storm, np.ndarray): continue
        lon_storm[lon_storm<0] = lon_storm[lon_storm<0] + 360

        # stack storm longitude, latitude
        lonlat_s = np.column_stack(
            (lon_storm, lat[i_storm])
        )

        # index for removing nans
        ix_nonan = ~np.isnan(lonlat_s).any(axis=1)
        lonlat_s = lonlat_s[ix_nonan]

        lon_ps, lat_ps = lonlat_s[:,0], lonlat_s[:,1]

        # check coordinates within the rectangle
        if ((lon_ps <= lon2) & (lon_ps >= lon1) & (lat_ps >= lat1) & (lat_ps <= lat2)).any():
            l_storms_area.append(i_storm)

    # cut storm dataset to selection
    xds_TCs_sel = xds_TCs.isel(storm=l_storms_area)
    xds_TCs_sel = xds_TCs_sel.assign_coords(storm = np.array(l_storms_area))

    return xds_TCs_sel

def dataframe_pressures_ibtracs(xds_ibtracs):
    '''
    From IBTrACS obtain a dataframe including:
        -id of the storm
        -longitude
        -latitude
        -pressure
        -time

    Up to 04/2018 using pressure from wmo agency and from this date
    using USA pressure since there is no more wmo pressure data

    At the end dropping the last tracks since the format is timestamp
    and not datetime since they have not been updated yet to datetime format.
    '''

    # WMO center
    # appended track data coordinates with pressure value
    storm, time, lon, lat, pres = [], [], [], [], []

    basin = xds_ibtracs.basin.values

    storms_SP = xds_ibtracs

    # # basin IDs (b'EP' b'NA' b'NI' b'SA' b'SI' b'SP' b'WP')
    # storms_SP = np.unique(np.where(basin == np.bytes_(b'SP'))[0])
    # storms_SI = np.unique(np.where(basin == np.bytes_(b'SI'))[0])

    # # remove intersect positions (SI basin)
    # _, common_ind, _ = np.intersect1d(storms_SP, storms_SI, return_indices=True)
    # storms_SP = np.delete(storms_SP, common_ind)

    for st_i in range(len(storms_SP.storm.values)):
        st = xds_ibtracs.isel(storm=st_i)

        # check for pressure values 
        p = st.wmo_pres.values
        pos = p[~np.isnan(p)]

        if pos.size > 0:
            # check longitude convention [360º], from -180 to 180 to 0 to 360
            st.lon.values[st.lon.values < 0] += 360

            # store variables
            storm.extend([st_i]*pos.size)
            time.extend(list(st.time.values[~np.isnan(p)]))
            lon.extend(list(st.lon.values[~np.isnan(p)]))
            lat.extend(list(st.lat.values[~np.isnan(p)]))
            pres.extend(list(p[~np.isnan(p)]))

    # store in dataframe
    df_wmo = pd.DataFrame({
        'st': np.asarray(storm),
        'time': np.asarray(time),
        'lon': np.asarray(lon),
        'lat': np.asarray(lat),
        'pres': np.asarray(pres),
    })

    # USA center
    # appended track data coordinates with pressure value
    storm, time, lon, lat, pres = [], [], [], [], []

    for st_i in range(len(storms_SP.storm.values)):
        st = xds_ibtracs.isel(storm=st_i)

        # check for pressure values 
        p = st.usa_pres.values
        pos = p[~np.isnan(p)]

        if pos.size > 0:
            # check longitude convention [360º], from -180 to 180 to 0 to 360
            st.lon.values[st.lon.values < 0] += 360

            # store variables
            storm.extend([st_i]*pos.size)
            time.extend(list(st.time.values[~np.isnan(p)]))
            lon.extend(list(st.lon.values[~np.isnan(p)]))
            lat.extend(list(st.lat.values[~np.isnan(p)]))
            pres.extend(list(p[~np.isnan(p)]))

    # store in dataframe
    df_usa = pd.DataFrame({
        'st': np.asarray(storm),
        'time': np.asarray(time),
        'lon': np.asarray(lon),
        'lat': np.asarray(lat),
        'pres': np.asarray(pres),
    })

    # keep only the USA pressure from April 2018 onwards
    df_usa = df_usa.iloc[6506:]

    # final df with wmo pressure from the beggining to 04/2018
    # and usa pressure from than moment onwards
    df0 = pd.concat([df_wmo,df_usa])

    # actualizar los índices para que vayan del 1 al final
    # y no se mantengan los de los anteriores df
    n_rows = df0.count()
    n_rows[0]
    arr = np.arange(0,n_rows[0])
    s = pd.Series(arr)
    df0 = df0.set_index([s])

    # trampilla para eliminar los últimos datos, que tienen todavía timestamps:
    # From february 2021 the dates are no longer datetime64 but timestamps, so the above cell code 
    # to loop to load yearly datasets and store nearest variable values no longer works and needs 
    # a modification to adequate the format of the dates from timestamp to datetime.
    df_n = df0.drop(df0.index[14614:14725])
    #df_cali = df0.drop(df0.index[14279:])
    n_rows = df_n.count()
    n_rows[0]
    arr = np.arange(0,n_rows[0])
    s = pd.Series(arr)
    df_now = df_n.set_index([s])

    return df_now

def add_sst_mld(df0, p_sst, p_mld, lo1, lo2, la1, la2):
    '''
    Function to add to the dataframe obtained from fucntion
    dataframe_pressures_ibtracs (it includes
    information of the id of the storm, longitude, latitude, pressure and time of
    the tc tracks included in IBTrACS) the mld and sst corresponding data from year
    1981 up to when tc track data ends.

    df0 -
    p_sst -
    p_mld -
    lo1 -
    lo2 -
    la1 -
    la2 -
    '''

    # TODO he cambiado seleciones loc por iloc

    print('Start time: ', datetime.now())

    # predictor area
    lo_area = [lo1, lo2]
    la_area = [la1, la2]

    year = pd.DatetimeIndex(df0['time']).year
    month = pd.DatetimeIndex(df0['time']).month

    sst_near, mld_near, pos = [], [], []
    for i in range(df0['time'].size):

        if (year[i] == 1981) & (month[i] >= 9) | (year[i] > 1981):

            # remove ns decimals to locate SST time
            time_df = df0['time'].values[i].astype('datetime64[D]').astype('datetime64[ns]')

            # open yearly files
            sst = xr.open_dataset(op.join(p_sst, 'sst.day.mean.{0}.nc'.format(year[i])), engine='netcdf4')
            sst_sel = sst.sel(time = time_df)

            mld = xr.open_dataset(op.join(p_mld, 'ocnmld.day.mean.{0}.nc'.format(year[i])), engine='netcdf4')
            mld_sel = mld.sel(time = time_df)

            # find nearest coordinates
            ind_lon_sst = np.argmin(abs(sst_sel.lon.values - df0.iloc[i]['lon']))
            ind_lat_sst = np.argmin(abs(sst_sel.lat.values - df0.iloc[i]['lat']))
            ind_lon_mld = np.argmin(abs(mld_sel.longitude.values - df0.iloc[i]['lon']))
            ind_lat_mld = np.argmin(abs(mld_sel.latitude.values - df0.iloc[i]['lat']))

            # find closest coordinate
            pos.append(i)
            sst_near.append(sst_sel.sst[ind_lat_sst, ind_lon_sst].values)
            mld_near.append(mld_sel.dbss[ind_lat_mld, ind_lon_mld].values)

    # select coordinates data with both SST and MLD
    df1 = df0.iloc[pos]

    # add new columns
    df1['sst'] = np.asarray(sst_near)
    df1['mld'] = np.asarray(mld_near)

    # drop NaNs
    df2 = df1.dropna(how='any', axis='rows')

    # filter by target area
    df = df2.loc[(df2['lon'] >= lo_area[0]) & (df2['lon'] <= lo_area[1]) & (df2['lat'] >= la_area[0])]

    print('End time:   ', datetime.now(), '\n')

    return df

def ds_index_sst_mld_calibration(p_sst, p_mld, df, lop, lop2, lap, lap2, delta):
    '''
    This Function takes the dataframe obtained from fucntion df_p_sst_mld (it includes
    information of the id of the storm, longitude, latitude, pressure, time of
    the tc tracks included in IBTrACS; mld and sst ).

    It computes the index values from the data in the dataframe, discretized in
    intervals of 0.5ºC and 0.5m for sst and mld respectively and builds a dataset for all
    the calibration period (1982-2019) that includes in the predictor grid
    the index, sst and mld data with longitude, latitude and time as dimensions

    p_sst -
    p_mld -
    df    -
    lop   -
    lop2  -
    lap   -
    lap2  -
    delta -
    '''


    # discretization: 0.5ºC (SST), 5m (MLD)
    xx,yy,zz = data_to_discret(df['sst'].values, df['mld'].values, 0.5, 5, df['pres'].values, 15, 32, 0, 175, option='min')

    # index function
    index = zz
    fmin, fmax = np.nanmin(zz), np.nanmax(zz)
    index_zz = (fmax - index) / (fmax-fmin)

    # remove nan
    index_zz[np.isnan(index_zz)] = 0

    # loop to process mld and sst data and joined it in a dataset     
    sst_ls, mld_ls = [], []

    # target   
    lo_sel = np.arange(lop, lop2, delta)
    la_sel = np.arange(lap, lap2, -delta)

    print('Start time: ', datetime.now())

    for i in np.arange(1982, 2019+1):

        sst = xr.open_dataset(op.join(p_sst, 'sst.day.mean.{0}.0505.nc'.format(i)))
        sst_ls.append(sst.sel(lat=la_sel, lon=lo_sel))

        mld = xr.open_dataset(op.join(p_mld,'ocnmld.day.mean.{0}.nc'.format(i)))
        mld_ls.append(mld.sel(latitude=la_sel, longitude=lo_sel))

    print('End   time: ', datetime.now())

    sst_merge = xr.merge(sst_ls)
    mld_merge = xr.merge(mld_ls)

    print('Merge time: ', datetime.now(), '\n')

    # obtain index over time
    var_x = sst_merge.sst.values
    var_y = mld_merge.dbss.values
    index_merge = np.zeros(sst_merge.sst.shape)

    #tamaño matriz index_zz (36x35)
    for i in range(xx.shape[0]-1):      # mld 36
        for j in range(xx.shape[1]-1):  # sst 35

            pos = np.where((var_x>=xx[0,j]) & (var_x<=xx[0,j+1]) & (var_y>=yy[i,0]) & (var_y<yy[i+1,0]))
            index_merge[pos] = index_zz[i,j]
            #print(pos)

    # return dataset
    xs = xr.Dataset(
        {
            'index': (('time','lat','lon',), index_merge),
            'sst': (('time','lat','lon',), sst_merge.sst.values),
            'dbss': (('time','lat','lon',), mld_merge.dbss.values),
        },
        {
            'time': sst_merge.time.values,
            'lat': sst_merge.lat.values,
            'lon': sst_merge.lon.values,
        })

    # add attributes
    xs.index.attrs['varname']='Index (sst-mld-pmin)'
    xs.index.attrs['units']=''
    xs.sst.attrs['varname']='Sea Surface Temperature'
    xs.sst.attrs['units']='ºC'
    xs.dbss.attrs['varname']='Mixed Layer Depth'
    xs.dbss.attrs['units']='m'

    # add mask [0:sea, 1:land]
    mask = np.zeros((xs.sst.values[0,:,:].shape))
    ind = np.isnan(xs.sst.values.sum(axis=0) + xs.dbss.values.sum(axis=0))
    mask[ind] = 1
    xs['mask'] = (('lat','lon',), mask)

    return xs

def ds_trmm(p_trmm, lop, lop2, lap, lap2, delta):
    '''
    It takes TRMM data, that starts in 1998 and puts it into a dataset with longitude
    and latitude as dimensions, in the predictor grid


    lop   -
    lop2  -
    lap   -
    lap2  -
    delta -
    '''

    # TRMM data starts from 1998
    # loop
    trmm_ls = []

    lo_sel = np.arange(lop,lop2, delta)
    la_sel = np.arange(lap,lap2, -delta)

    print('Start time: ', datetime.now())
    for i in np.arange(1998, 2019+1):

        trmm = xr.open_dataset(op.join(p_trmm, 'yearly/3B42.precipitation.day.accum.{0}.05.nc'.format(i)))
        trmm_ls.append(trmm.sel(lat=la_sel, lon=lo_sel))

    print('End   time: ', datetime.now())

    trmm_merge = xr.merge(trmm_ls)

    print('Merge time: ', datetime.now(), '\n')

    trmm_merge.precipitation.attrs['varname'] = 'Daily precipitation'
    trmm_merge.precipitation.attrs['units'] = 'mm'

    return trmm_merge

# TODO dwt_tcs_count y dwt_tcs_count_tracks duplicados?
def dwt_tcs_count(xds, df, categ=1100, dx=2, dy=2, lo0=None, lo1=None, la0=None, la1=None):
    '''
    Build the dataset with the resulting counting from df_storm_counts function.

    Inputs:
        -the dataset with the resulting pca-kmeans classification (xds)
        -the df with the lon,lat,time,id storm, sst and mld data for the calibration period
        -cells size dxº x dyº
        -longitude limits of the targe area (lo0,lo1)
        -latitude limits of the target area(la0,la1)
        -category threshold for the TCs (categ)

    Output, the dataset:
        -the absolute total number of segments per cell per DWT (cnt_tcs)
        -the total number of days per DWT (cnt_dwts)
        -the number of segments per cell per DWT in daily basis (prob_tcs_dwts)
    '''

    if not (np.array((lo0, lo1, la0, la1))).any():
        lo0, lo1 = xds.lon.min(), xds.lon.max()
        la0, la1 = xds.lat.min(), xds.lat.max()

    bmus = xds.bmus.values
    lo = np.arange(lo0, lo1+dx, dx)
    la = np.arange(la0, la1+dy, dy)

    st_count = np.zeros((xds.n_clusters.shape[0], la.shape[0], lo.shape[0])) * np.nan #[]
    dwt_count = np.zeros(xds.n_clusters.shape[0]) * np.nan
    prob_dwts = np.zeros((xds.n_clusters.shape[0], la.shape[0], lo.shape[0])) * np.nan #[]

    # each DWT
    for i in range(xds.n_clusters.values.shape[0]):

        # select dates
        pos = np.where(bmus == i)
        dates = xds.time.values[pos]
        DD = pd.DatetimeIndex(dates).day
        MM = pd.DatetimeIndex(dates).month
        YY = pd.DatetimeIndex(dates).year

        # count of DWT days
        dwt_number_i, dwt_count_i = np.unique(np.stack((YY, MM, DD)), return_counts=True, axis=1)
        dwt_count[i] = dwt_count_i.shape[0]

        # count of single storms (discretization dxdy)
        st_count[i,:,:] = df_storm_counts(dates, df, dx, dy, lo0, lo1, la0, la1, categ=categ)

        # probability of counts & DWT
        prob_dwts[i,:,:] = st_count[i,:,:] / dwt_count[i]

    # store dataset
    xs_cnt_dwt = xr.Dataset(
        {
            'cnt_tcs': (('n_clusters','lat','lon'), st_count),          # count storms per cell
            'cnt_dwts': (('n_clusters'), dwt_count),                    # count days per dwt
            'prob_tcs_dwts': (('n_clusters','lat','lon'), prob_dwts),   # num storms / days dwt
        },
        {
            'lat': la,
            'lon': lo,
        },
    )

    return xs_cnt_dwt

def df_storm_counts_tracks(dates, df_filtered, dx, dy, lo0, lo1, la0, la1, categ=1100):
    '''
    counts of SINGLE storms over discretized cells selecting according
    to the date of the daily segment with minimum pressure of the cell
    dates: corresponding to DWTi

    'df_filtered': track coordinates within the target area defined

    discretization in cells of dxº x dyº
    lo0, lo1 = longitude limits of the target area
    la0, la1 = latitude limits of the target area
    categ = category threshold for the TCs
    '''

    # category filter
    cat_filter = categ # 979    # category 2 or higher

     # counter of (unique) tracks per cell 
    lo_ = np.arange(lo0, lo1 + dx, dx)
    la_ = np.arange(la0, la1 + dy, dy)
    st_count = np.zeros((la_.shape[0], lo_.shape[0])) * np.nan    
    
    for j,clo in enumerate(lo_[:-1]):
        for i,cla in enumerate(la_[:-1]):

            df_cell = df_filtered.loc[
                (df_filtered['lon'] >= clo) & (df_filtered['lon'] < clo+dx) & 
                (df_filtered['lat'] >= cla) & (df_filtered['lat'] < cla+dy) &
                (df_filtered['pres'] <= cat_filter)
            ]

            ids = df_cell['st'].values
            ids_uni = np.unique(ids)
            st_t_list = []

            for k in range(ids_uni.shape[0]):
                df_track = df_cell.loc[(df_cell['st'] == ids_uni[k])]
                prs = df_track['pres'].values
                min_pos = np.argmin(prs)
                all_dates = df_track['time'].values.astype('datetime64[D]')
                date_min_track = all_dates[min_pos]
                st_t_list.append(date_min_track)

            st_t = np.array(st_t_list)

            if st_t.shape[0] > 0:
                common_t, ind1, _ = np.intersect1d(st_t, dates, return_indices=True)


    #         if ind1.shape[0] > 0:
    #             DD = pd.DatetimeIndex(df_cell['time'].values[ind1]).day
    #             MM = pd.DatetimeIndex(df_cell['time'].values[ind1]).month
    #             YY = pd.DatetimeIndex(df_cell['time'].values[ind1]).year
    #             st_id = df_cell['st'].values[ind1]


    #             #remove repetitions
    #             st_number_ij, st_count_ij = np.unique(np.stack((st_id, YY, MM, DD)), return_counts=True, axis=1)
                if common_t.shape[0]>0:
                    st_count[i,j] = common_t.shape[0]

    return st_count

def dwt_tcs_count_tracks(xds, df, categ=1100,dx=2, dy=2, lo0=None, lo1=None, la0=None, la1=None):
    '''
    Build the dataset with the resulting counting from df_storm_counts_tracks function.

    Inputs:
        -the dataset with the resulting pca-kmeans classification (xds)
        -the df with the lon,lat,time,id storm, sst and mld data for the calibration period (df)
        -cells size dxº x dyº
        -longitude limits of the targe area (lo0,lo1)
        -latitude limits of the target area(la0,la1)
        -category threshold for the TCs (categ)

    Output, the dataset:
        -the absolute total number of tracks per cell per DWT (cnt_tcs)
        -the total number of days per DWT (cnt_dwts)
        -the number of tracks per cell per DWT in daily basis (prob_tcs_dwts)  

    '''

    if not (np.array((lo0, lo1, la0, la1))).any():
        lo0, lo1 = xds.lon.min(), xds.lon.max()
        la0, la1 = xds.lat.min(), xds.lat.max()

    bmus = xds.bmus.values
    lo = np.arange(lo0, lo1+dx, dx)
    la = np.arange(la0, la1+dy, dy)

    st_count = np.zeros((xds.n_clusters.shape[0], la.shape[0], lo.shape[0])) * np.nan #[]
    dwt_count = np.zeros(xds.n_clusters.shape[0]) * np.nan
    prob_dwts = np.zeros((xds.n_clusters.shape[0], la.shape[0], lo.shape[0])) * np.nan #[]

    # each DWT
    for i in range(xds.n_clusters.values.shape[0]):
        # select dates
        pos = np.where(bmus == i)
        dates = xds.time.values[pos]
        DD = pd.DatetimeIndex(dates).day
        MM = pd.DatetimeIndex(dates).month
        YY = pd.DatetimeIndex(dates).year

        # count of DWT days
        dwt_number_i, dwt_count_i = np.unique(np.stack((YY, MM, DD)), return_counts=True, axis=1)
        dwt_count[i] = dwt_count_i.shape[0]

        # count of single storms (discretization dxdy)
        st_count[i,:,:] = df_storm_counts_tracks(dates, df, dx, dy, lo0, lo1, la0, la1, categ=categ)

        # probability of counts & DWT
        prob_dwts[i,:,:] = st_count[i,:,:] / dwt_count[i]

        # store dataset
    xs_cnt_dwt = xr.Dataset(
        {
            'cnt_tcs': (('n_clusters','lat','lon'), st_count),          # count storms per cell
            'cnt_dwts': (('n_clusters'), dwt_count),                    # count days per dwt
            'prob_tcs_dwts': (('n_clusters','lat','lon'), prob_dwts),   # num storms / days dwt
        },
        {
            'lat': la,
            'lon': lo,
        },
    )

    return xs_cnt_dwt

def ds_timeline(df, xs_dwt_counts, xs_dwt_counts_964, xds_kma):
    '''
    From the storm counting over the calibration period (xs_dwt_counts and xs_dwt_counts_964),
    the historical tracs (df) and the k-means fitted, construct a dataset for
    all the calibration period for the target area in a daily basis with the following data:

    -DWTs along time
    -id_tcs for tcs day
    -mask_tcs to distinguish tcs day 

    *Depending on the info on xs_dwt_counts (all TCs) / xs_dwt_counts_964 (TCs filtering from category 3) 
    we are counting segments of tracks 
    -the absolute total number of tracks/segments per cell per DWT (cnt_tcs and counts_tcs_964)
    -the mean expected number of tracks/segments/ or  TC probability per cell per DWT in daily basis (prob_tcs_dwts)
    '''

    # generate daily mask (TCs, counts, probs)
    dateline = pd.DatetimeIndex(xds_kma.time.values.astype('datetime64[D]'))
    st_id = df['st'].values
    DD = pd.DatetimeIndex(df['time'].values).day
    MM = pd.DatetimeIndex(df['time'].values).month
    YY = pd.DatetimeIndex(df['time'].values).year

    # daily mask for tcs
    mask_tcs_D = np.zeros(dateline.shape, dtype=bool)
    id_tcs_D = np.zeros(dateline.shape, dtype=list)*np.nan

    # dwts counts, probs
    counts_tcs_D = np.zeros((dateline.size, xs_dwt_counts.lat.size, xs_dwt_counts.lon.size))*np.nan
    probs_tcs_D = np.zeros((dateline.size, xs_dwt_counts.lat.size, xs_dwt_counts.lon.size))*np.nan

    #dwts counts, probs>= category 3
    counts_tcs_D_val_964 = np.zeros((dateline.size, xs_dwt_counts_964.lat.size, xs_dwt_counts_964.lon.size))*np.nan
    probs_tcs_D_val_964 = np.zeros((dateline.size, xs_dwt_counts_964.lat.size, xs_dwt_counts_964.lon.size))*np.nan


    for pos, idate in enumerate(dateline):

        pos_tcs = np.where((YY==idate.year) & (MM==idate.month) & (DD==idate.day))[0]
        ibmus = xds_kma.bmus.values[pos]

        if pos_tcs.any():
            mask_tcs_D[pos] = True
            id_tcs_D[pos] = np.unique(st_id[pos_tcs])#np.array(list(st_id[pos_tcs]))

        counts_tcs_D[pos,:,:] = xs_dwt_counts.cnt_tcs.values[ibmus,:,:]
        probs_tcs_D[pos,:,:] = xs_dwt_counts.prob_tcs_dwts.values[ibmus,:,:]

        counts_tcs_D_val_964[pos,:,:] = xs_dwt_counts_964.cnt_tcs.values[ibmus,:,:]
        probs_tcs_D_val_964[pos,:,:] = xs_dwt_counts_964.prob_tcs_dwts.values[ibmus,:,:]


    # return dataset
    xds_timeline = xr.Dataset({
        'bmus': (('time'), xds_kma.bmus.values),
        'mask_tcs': (('time'), mask_tcs_D),
        'id_tcs': (('time'), id_tcs_D),
        'counts_tcs': (('time','lat','lon'), counts_tcs_D),
        'counts_tcs_964': (('time','lat','lon'), counts_tcs_D_val_964),    
        'probs_tcs': (('time','lat','lon'), probs_tcs_D),
        'probs_tcs_964': (('time','lat','lon'), probs_tcs_D_val_964),
    },{
        'time': xds_kma.time.values,
        'lat': xs_dwt_counts.lat.values,
        'lon': xs_dwt_counts.lon.values,
    })

    return xds_timeline

def variables_dwt_super_plot(xds_kma, xds_timeline):
    '''
    Obtain the days and tcs days for each year according to the DWT 
    to plot in the historical chronology plot
    '''

    # reshape [years,days]
    year = pd.DatetimeIndex(xds_kma.time.values).year
    dateline = pd.DatetimeIndex(xds_kma.time.values.astype('datetime64[D]'))
    mask_bmus_YD = np.zeros((np.unique(year).shape[0], 366))*np.nan
    mask_tcs_YD = np.zeros((np.unique(year).shape[0], 366), dtype=bool)
    mask_ids_YD = np.zeros((np.unique(year).shape[0], 366), dtype=list)*np.nan
    probs_sum_YD = np.zeros((np.unique(year).shape[0], 366))
    probs_max_YD = np.zeros((np.unique(year).shape[0], 366))
    probs_mean_YD = np.zeros((np.unique(year).shape[0], 366))

    pos = 0
    for i, iyear in enumerate(np.unique(dateline.year)):
        for j in range(366+1):

            if dateline.year[pos] == iyear:
                mask_bmus_YD[i,j] = xds_timeline.bmus.values[pos]
                mask_tcs_YD[i,j] = xds_timeline.mask_tcs.values[pos]
                mask_ids_YD[i,j] = xds_timeline.id_tcs.values[pos]
                probs_sum_YD[i,j] = np.nansum(xds_timeline.probs_tcs[pos,:,:])
                probs_max_YD[i,j] = np.nanmax(xds_timeline.probs_tcs[pos,:,:])
                probs_mean_YD[i,j] = np.nanmean(xds_timeline.probs_tcs[pos,:,:])

                # break final loop
                if pos < dateline.size-1:     pos += 1
                elif pos == dateline.size-1:  break

    return mask_bmus_YD, mask_tcs_YD


def ds_index_sst_mld_slp_pp_calibration(path_slp,xs,lop,lop2,lap,lap2,delta):    
     
    '''    
    It takes the dataset going out from ds_index_sst_mld_calibration, that includes 
    in the predictor grid the index, sst and mld data with longitude, latitude and time as dimensions
    for the calibration period (1982-2019) and adds to it the SLP and precipitation data'''
    
    # grid resolution
     
    # target
    lo_sel = np.arange(lop,lop2, delta)
    la_sel = np.arange(lap,lap2, -delta)
    slp_ls, pratel_ls = [], []
    print('Start time: ', datetime.now())

    for i in np.arange(1982, 2019+1):

        slp = xr.open_dataset(path_slp+'/prmsl/prmsl.day.mean.{0}.05.nc'.format(i))
        slp_ls.append(slp.sel(lat=la_sel, lon=lo_sel))


    print('End   time: ', datetime.now())

    slp_merge = xr.merge(slp_ls)

    print('Merge time: ', datetime.now(), '\n')
    
    # add to dataset
    xs['slp'] = (('time','lat','lon',), slp_merge.sst.values)

    # add attributes
    xs.slp.attrs['varname']='Pressure'
    xs.slp.attrs['units']='Pa'
    
    return xs