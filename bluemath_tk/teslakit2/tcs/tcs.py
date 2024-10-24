#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import radians, degrees, sin, cos, asin, acos, sqrt, atan2, pi
import numpy as np
import xarray as xr

from ..util.operations import GetUniqueRows


def GeoDistance(lat1, lon1, lat2, lon2):
    'Returns great circle distance between points in degrees'

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2;
    if a < 0: a = 0
    if a > 1: a = 1

    r = 1
    rng = r * 2 * atan2(sqrt(a), sqrt(1-a))
    rng = degrees(rng)

    return rng

def GeoAzimuth(lat1, lon1, lat2, lon2):
    'Returns geodesic azimuth between point1 and point2'

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    az = atan2(
        cos(lat2) * sin(lon2-lon1),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2-lon1)
    )
    if lat1 <= -pi/2: az = 0
    if lat2 >=  pi/2: az = 0
    if lat2 <= -pi/2: az = pi
    if lat1 >=  pi/2: az = pi

    az = az % (2*pi)
    az = degrees(az)

    return az


def Extract_Circle(xds_TCs, p_lon, p_lat, r, d_vns):
    '''
    Extracts TCs inside circle - used with NWO or Nakajo databases

    xds_TCs: tropical cyclones track database
        lon, lat, pressure variables
        storm dimension

    circle defined by:
        p_lon, p_lat  -  circle center
        r             -  circle radius (degree)

    d_vns: dictionary to set longitude, latitude, time and pressure varnames

    returns:
        xds_area: selection of xds_TCs inside circle
        xds_inside: contains TCs custom data inside circle
    '''

    # point longitude and latitude
    lonlat_p = np.array([[p_lon, p_lat]])

    # get names of vars: longitude, latitude, pressure and time
    nm_lon = d_vns['longitude']
    nm_lat = d_vns['latitude']
    nm_prs = d_vns['pressure']
    nm_tim = d_vns['time']

    # storms longitude, latitude, pressure and time (if available)
    lon = xds_TCs[nm_lon].values[:]
    lat = xds_TCs[nm_lat].values[:]
    prs = xds_TCs[nm_prs].values[:]
    time = xds_TCs[nm_tim].values[:]

    # get storms inside circle area
    n_storms = xds_TCs.storm.shape[0]
    l_storms_area = []

    # inside parameters holders
    l_prs_min_in = []   # circle minimun pressure
    l_prs_mean_in = []  # circle mean pressure
    l_vel_mean_in = []  # circle mean translation speed
    l_categ_in = []     # circle storm category
    l_date_in = []      # circle date (day)
    l_date_last = []    # last cyclone date 
    l_gamma = []        # azimuth 
    l_delta = []        # delta 

    l_ix_in = []        # historical enters the circle index
    l_ix_out = []       # historical leaves the circle index 

    for i_storm in range(n_storms):

        # fix longitude <0 data and skip "one point" tracks
        lon_storm = lon[i_storm]
        if not isinstance(lon_storm, np.ndarray): continue
        lon_storm[lon_storm<0] = lon_storm[lon_storm<0] + 360

        # stack storm longitude, latitude
        lonlat_s = np.column_stack(
            (lon_storm, lat[i_storm])
        )

        # index for removing nans
        ix_nonan = ~np.isnan(lonlat_s).any(axis=1)
        lonlat_s = lonlat_s[ix_nonan]

        # remove storms with only one track point
        if len(lonlat_s)<=1:
            continue

        # calculate geodesic distance (degree)
        geo_dist = []
        for lon_ps, lat_ps in lonlat_s:
            geo_dist.append(GeoDistance(lat_ps, lon_ps, p_lat, p_lon))
        geo_dist = np.asarray(geo_dist)

        # find storm inside circle and calculate parameters
        if (geo_dist < r).any():

            # storm inside circle
            ix_in = np.where(geo_dist < r)[0][:]

            # storm translation velocity
            geo_dist_ss = []
            for i_row in range(lonlat_s.shape[0]-1):
                i0_lat, i0_lon = lonlat_s[i_row][1], lonlat_s[i_row][0]
                i1_lat, i1_lon = lonlat_s[i_row+1][1], lonlat_s[i_row+1][0]
                geo_dist_ss.append(GeoDistance(i0_lat, i0_lon, i1_lat, i1_lon))
            geo_dist_ss = np.asarray(geo_dist_ss)

            # get delta time in hours (irregular data time delta)
            if isinstance(time[i_storm][0], np.datetime64):
                # round to days
                time[i_storm] = np.array(
                    [np.datetime64(xt, 'h') for xt in time[i_storm]]
                )

                delta_h = np.diff(
                    time[i_storm][~np.isnat(time[i_storm])]
                ).astype('timedelta64[h]').astype(float)

            else:
                # nakajo: time already in hours
                delta_h = np.diff(
                    time[i_storm][~np.isnan(time[i_storm])]
                ).astype(float)

            vel = geo_dist_ss * 111.0/delta_h  # km/h

            # promediate vel 
            velpm = (vel[:-1] + vel[1:])/2
            velpm = np.append(vel[0], velpm)
            velpm = np.append(velpm, vel[-1])

            # calculate azimuth 
            lat_in_end, lon_in_end = lonlat_s[ix_in[-1]][1], lonlat_s[ix_in[-1]][0]
            lat_in_ini, lon_in_ini = lonlat_s[ix_in[0]][1], lonlat_s[ix_in[0]][0]
            gamma = GeoAzimuth(lat_in_end, lon_in_end, lat_in_ini, lon_in_ini)
            if gamma < 0.0: gamma += 360

            # calculate delta
            nd = 1000
            st = 2*np.pi/nd
            ang = np.arange(0, 2*np.pi + st, st)
            xps = r * np.cos(ang) + p_lat
            yps = r * np.sin(ang) + p_lon
            angle_radius = []
            for x, y in zip(xps, yps):
                angle_radius.append(GeoAzimuth(lat_in_end, lon_in_end, x, y))
            angle_radius = np.asarray(angle_radius)

            im = np.argmin(np.absolute(angle_radius - gamma))
            delta = GeoAzimuth(p_lat, p_lon, xps[im], yps[im]) # (-180, +180)
            if delta < 0.0: delta += 360

            # more parameters 
            prs_s_in = prs[i_storm][ix_in]  # pressure

            # nan data filter
            if np.all(np.isnan(prs_s_in)):
                continue
            no_nan = ~np.isnan(prs_s_in)

            prs_s_in = prs_s_in[no_nan]
            prs_s_min = np.min(prs_s_in)  # pressure minimun
            prs_s_mean = np.mean(prs_s_in)

            vel_s_in = velpm[ix_in][no_nan]  # velocity
            vel_s_mean = np.mean(vel_s_in) # velocity mean

            categ = GetStormCategory(prs_s_min)  # category

            dist_in = geo_dist[ix_in][no_nan]
            p_dm = np.where((dist_in==np.min(dist_in)))[0]  # closest to point

            time_s_in = time[i_storm][ix_in][no_nan]  # time
            time_closest = time_s_in[p_dm][0]  # time closest to point 

            # filter storms 
            # TODO: storms with only one track point inside radius. solve?
            if np.isnan(np.array(prs_s_in)).any() or \
               (np.array(prs_s_in) <= 860).any() or \
               gamma == 0.0:
                continue

            # store parameters
            l_storms_area.append(i_storm)
            l_prs_min_in.append(np.array(prs_s_min))
            l_prs_mean_in.append(np.array(prs_s_mean))
            l_vel_mean_in.append(np.array(vel_s_mean))
            l_categ_in.append(np.array(categ))
            l_date_in.append(time_closest)
            l_gamma.append(gamma)
            l_delta.append(delta)

            # store historical indexes inside circle 
            l_ix_in.append(ix_in[0])
            l_ix_out.append(ix_in[-1])

            # store last cyclone date too
            l_date_last.append(time[i_storm][ix_nonan][-1])

    # cut storm dataset to selection
    xds_TCs_sel = xds_TCs.isel(storm=l_storms_area)
    xds_TCs_sel = xds_TCs_sel.assign_coords(storm = np.array(l_storms_area))

    # store storms parameters 
    xds_TCs_sel_params = xr.Dataset(
        {
            'pressure_min':(('storm'), np.array(l_prs_min_in)),
            'pressure_mean':(('storm'), np.array(l_prs_mean_in)),
            'velocity_mean':(('storm'), np.array(l_vel_mean_in)),
            'gamma':(('storm'), np.array(l_gamma)),
            'delta':(('storm'), np.array(l_delta)),
            'category':(('storm'), np.array(l_categ_in)),
            'dmin_date':(('storm'), np.array(l_date_in)),
            'last_date':(('storm'), np.array(l_date_last)),
            'index_in':(('storm'), np.array(l_ix_in)),
            'index_out':(('storm'), np.array(l_ix_out)),
        },
        coords = {
            'storm':(('storm'), np.array(l_storms_area))
        },
        attrs = {
            'point_lon' : p_lon,
            'point_lat' : p_lat,
            'point_r' : r,
        }
    )

    return xds_TCs_sel, xds_TCs_sel_params


def GetStormCategory(pres_min):
    '''
    Returns storm category (int 5-0)
    '''

    pres_lims = [920, 944, 964, 979, 1000]

    if pres_min <= pres_lims[0]:
        return 5
    elif pres_min <= pres_lims[1]:
        return 4
    elif pres_min <= pres_lims[2]:
        return 3
    elif pres_min <= pres_lims[3]:
        return 2
    elif pres_min <= pres_lims[4]:
        return 1
    else:
        return 0


def ibtrac_basin_fitting(x0, y0):
    '''
    Assigns cubic polynomial fitting curve coefficient for each basin of
    historical TCs data (IBTrACS)
    '''

    # determination of the location basin
    if y0 < 0:                  basin = 5
    elif (y0 > 0) & (x0 > 0):   basin = 3
    else:                       print('Basin not defined')

    # cubic polynomial fitting curve for Ibtracs and each basin
    # TODO: obtain all basin fitting coefficients

    if basin == 3:      # West Pacific
        p1 = -7.77328602747578e-06
        p2 = 0.0190830514629838
        p3 = -15.9630945598490
        p4 = 4687.76462404360

    elif basin == 5:    # South Pacific
        p1 = -4.70481986864773e-05
        p2 = 0.131052968357409
        p3 = -122.487981649828
        p4 = 38509.7575283218

    return p1, p2, p3, p4


def filter_simultaneous_TCs(TCs_r1_params):

    '''
     For TCs where time closest to site coincides on the same day or consecutive days,
     keep TCs of greater category or longest track inside circle
    '''

    # sort TCs by date
    TCs_r1_params = TCs_r1_params.sortby(TCs_r1_params.dmin_date)

    while True:
        storm_cat = TCs_r1_params.category.values[:]
        storm_dates = TCs_r1_params.dmin_date.values[:]
        index_out = TCs_r1_params.index_out.values[:] # length of the TC inside the circle

        # obtain time differences
        t_dif = []
        for d in range(len(storm_dates)-1):
            t_dif.append(np.timedelta64(storm_dates[d+1]-storm_dates[d], 'D'))

        # consecutive TCs
        ind_1d = np.where(t_dif<=np.timedelta64(1,'D'))[0]
        if len(ind_1d) == 0:
            break

        # Choose between simultaneous TCs
        ind_sel = np.arange(0,len(storm_dates))
        ind_sel = ind_sel.astype("float")
        for d in ind_1d:

            if storm_cat[d] > storm_cat[d+1]:
                ind_sel[d+1] = np.nan

            elif storm_cat[d] < storm_cat[d+1]:
                ind_sel[d] = np.nan

            else:
                if index_out[d] >= index_out[d+1]:
                    ind_sel[d+1] = np.nan

                if index_out[d] < index_out[d+1]:
                    ind_sel[d] = np.nan

        ind_sel = ind_sel[~np.isnan(ind_sel)]
        ind_sel = ind_sel.astype("int")


        # select TCs
        TCs_r1_params = TCs_r1_params.isel(storm = ind_sel, drop=True)


    return TCs_r1_params
