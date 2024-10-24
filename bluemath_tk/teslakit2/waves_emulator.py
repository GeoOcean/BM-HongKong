import numpy as np
import xarray as xr
import time
from datetime import datetime
from .io.aux_nc import StoreBugXdset
import os.path as op
from netCDF4 import Dataset


def swells_emulator(n, DWTs_sim, n_swells, snakes_sim, max_num_snakes, p_out_waves):

    start = time.time()

    # define output
    xds_wvs_sim = DWTs_sim.copy(deep=True)
    xds_wvs_sim['Hs']=(('time','n_snake'), np.zeros((len(DWTs_sim.time),  max_num_snakes))*np.nan)
    xds_wvs_sim['Tp']=(('time','n_snake'), np.zeros((len(DWTs_sim.time),  max_num_snakes))*np.nan)
    xds_wvs_sim['Dir']=(('time','n_snake'), np.zeros((len(DWTs_sim.time), max_num_snakes))*np.nan)
    xds_wvs_sim['Duration']=(('time','n_snake'), np.zeros((len(DWTs_sim.time), max_num_snakes))*np.nan)
    xds_wvs_sim['Tau']=(('time','n_snake'), np.zeros((len(DWTs_sim.time),  max_num_snakes))*np.nan)
    xds_wvs_sim['sh1']=(('time','n_snake'), np.zeros((len(DWTs_sim.time),  max_num_snakes))*np.nan)
    xds_wvs_sim['sf1']=(('time','n_snake'), np.zeros((len(DWTs_sim.time),  max_num_snakes))*np.nan)
    xds_wvs_sim['H_ini']=(('time','n_snake'), np.zeros((len(DWTs_sim.time),  max_num_snakes))*np.nan)


    # for each bmus, select number of probable swells and parameters (for each direction)
    for ind_dd, dd in enumerate(DWTs_sim.evbmus_sims.values):
        dd = int(dd)-1 # bmus from 0-41 instead of 1-42

        if dd>35:
            continue

        # select one random day for that dwt:
        n_swells_dd = n_swells.where(n_swells.bmus==dd, drop=True)
        int_rnd = np.random.randint(0,len(n_swells_dd.time))
        n_swells_rnd = n_swells_dd.isel(time=int_rnd)

        # select directions that contain snakes for that day
        ind_dir_snakes = np.where(n_swells_rnd.n_swells!=0)[0]

        if len(ind_dir_snakes)>0:

            cont=0
            for i in (ind_dir_snakes):

                # select snakes for that direction:
                snakes_sim_dir = snakes_sim.isel(dire=i, wt=dd)

                # for the number of snakes, select random parameters:
                for j in range(int(n_swells_rnd.n_swells[i].values)):

                    # select one of the copula sims:
                    int_rnd = np.random.randint(0,len(snakes_sim.sim))
                    snakes_sim_rnd = snakes_sim_dir.isel(sim=int_rnd)

                    # save
                    xds_wvs_sim['Hs'][ind_dd,cont] = snakes_sim_rnd.Hs.values
                    xds_wvs_sim['Tp'][ind_dd,cont] = snakes_sim_rnd.Tp.values
                    xds_wvs_sim['Dir'][ind_dd,cont] = snakes_sim_rnd.Dir.values
                    xds_wvs_sim['Duration'][ind_dd,cont] = snakes_sim_rnd.Duration.values
                    xds_wvs_sim['Tau'][ind_dd,cont] = snakes_sim_rnd.Tau.values
                    xds_wvs_sim['sh1'][ind_dd,cont] = snakes_sim_rnd.sh1.values
                    xds_wvs_sim['sf1'][ind_dd,cont] = snakes_sim_rnd.sf1.values
                    xds_wvs_sim['H_ini'][ind_dd,cont] = snakes_sim_rnd.H_ini.values

                    cont=cont+1

    return xds_wvs_sim


def seas_emulator(DWTs_sim, seas_sim):

    # define output
    xds_wvs_sim = DWTs_sim.copy(deep=True)
    xds_wvs_sim['Hs_sea']=(('time'), np.full([len(DWTs_sim.time)],np.nan))
    xds_wvs_sim['Tp_sea']=(('time'), np.full([len(DWTs_sim.time)],np.nan))
    xds_wvs_sim['Dir_sea']=(('time'), np.full([len(DWTs_sim.time)],np.nan))


    # for each bmus, select number of probable swells and parameters
    for ind_dd, dd in enumerate(DWTs_sim.evbmus_sims.values):
        dd = int(dd)-1 # bmus from 0-41 instead of 1-42

        aa = DWTs_sim.evbmus_sims_awt.values[ind_dd]
        aa = int(aa)-1 # bmus from 0-5 instead of 1-6

        if dd>35:
            continue

        # select one of the random simulations for that dwt
        sim_rnd = np.random.randint(0, len(seas_sim.sim))
        seas_sim_dd = seas_sim.isel(bmus=dd, awt=aa, sim=sim_rnd)
        #seas_sim_dd = seas_sim.isel(bmus=dd, sim=sim_rnd)

        # save
        xds_wvs_sim['Hs_sea'][ind_dd] = seas_sim_dd.hs.values
        xds_wvs_sim['Tp_sea'][ind_dd] = seas_sim_dd.tp.values
        xds_wvs_sim['Dir_sea'][ind_dd] = seas_sim_dd.dpm.values

    return xds_wvs_sim


def read_hourly_sim(file, use_cftime):

    # netcdf4 lib
    fr = Dataset(file, 'r', format='NETCDF4')

    # get time (raw)
    v_id = fr.variables['time']
    v_a_t = dict([(k, v_id.getncattr(k)) for k in v_id.ncattrs()])
    v_v_t = v_id[:]

    # generate xarray dataset
    variables = list(fr.variables.keys())
    variables.remove('time')
    xds = xr.Dataset(coords = {'time': v_v_t})
    for var in variables:
        xds[var] = (('time'), fr.variables[var][:])

    xds['time'].attrs = v_a_t

    # close file
    fr.close()

    # optional decode times to np.datetime64 or DatetimeGregorian
    if use_cftime:
        xds = xr.decode_cf(xds, use_cftime=use_cftime)

    return xds


def add_TC_to_historical(spec_TC, spec, TC_cat):

    # change nans to 0
    spec_TC = spec_TC.where(~np.isnan(spec_TC.efth),0)

    # Downsample to spectrum coordinates
    spec_TC = spec_TC.drop(('lon','lat'))
    spec_TC = spec_TC.interp(coords={'dir':spec.dir, 'freq':spec.freq})

    # time of TC Hsmax
    spec_TC_max = np.sum(spec_TC.efth,axis=1)
    spec_TC_max = np.sum(spec_TC_max,axis=1)
    ind_no0 = np.where(spec_TC_max!=0)[0]    # drop 0 energy
    spec_TC = spec_TC.isel(time=ind_no0)
    spec_TC_max = spec_TC_max[ind_no0]
    ind_max = np.argmax(spec_TC_max.values)

    # get spectrum position for TC Hsmax
    spec_t = spec.efth.sel(time=spec_TC.time.isel(time=ind_max).values)
    spec_t = spec.efth.where(spec.time==spec_t.time)
    spec_t = spec_t.isel(dir=0,freq=0)
    ind_t = np.where(~np.isnan(spec_t))[0][0]

    # Substitute regular climate spectrum with TCs spectrum
    spec['efth'][ind_t-ind_max:ind_t+len(spec_TC_max)-ind_max,:,:] = spec_TC.efth.values
    spec['TC_cat'][ind_t-ind_max:ind_t+len(spec_TC_max)-ind_max]  = TC_cat.category
    spec['TC_id'][ind_t-ind_max:ind_t+len(spec_TC_max)-ind_max]  = TC_cat.id

    return spec


def add_TC_to_emulator(t, spec_TC, spec, TC_cat):

    # change nans to 0
    spec_TC = spec_TC.where(~np.isnan(spec_TC.efth),0)

    # Downsample to spectrum coordinates
    spec_TC = spec_TC.drop(('lon','lat'))
    spec_TC = spec_TC.interp(coords={'dir':spec.dir, 'freq':spec.freq})
    spec_TC = spec_TC.transpose('dir', 'freq', 'time')

    # time of TC Hsmax
    spec_TC_max = np.sum(spec_TC.efth,axis=0)
    spec_TC_max = np.sum(spec_TC_max,axis=0)
    ind_max = np.argmax(spec_TC_max.values)

    # get spectrum position for the day at 00:00h
    spec_t = spec.efth.sel(time=t)
    spec_t = spec.efth.where(spec.time==spec_t.time)
    spec_t = spec_t.isel(dir=0,freq=0)
    ind_t = np.where(~np.isnan(spec_t))[0][0]

    # get random hour position within that day where TC Hsmax is going to happen
    ind_t = np.random.randint(ind_t,ind_t+24)

    # Add TCs spectrum to regular climate spectrum
    # TODO: al inicio y final de año se corta el TC
    if (ind_t-ind_max)<0: # inicio
        print('TC not included')
    elif (ind_t+len(spec_TC_max)-ind_max)>len(spec.time): # fin
        print('TC not included')
    else:
        spec['efth'][:,:, ind_t-ind_max:ind_t+len(spec_TC_max)-ind_max] = spec.efth[:,:, ind_t-ind_max:ind_t+len(spec_TC_max)-ind_max] + spec_TC.efth.values
        spec['TC_cat'][ind_t-ind_max:ind_t+len(spec_TC_max)-ind_max]  = TC_cat.TC_cat.values
        spec['TC_type'][ind_t-ind_max:ind_t+len(spec_TC_max)-ind_max] = TC_cat.TC_type.values
        spec['TC_id'][ind_t-ind_max:ind_t+len(spec_TC_max)-ind_max]  = TC_cat.TC_id.values

    return spec
