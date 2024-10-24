#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op

# pip
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
from scipy.interpolate import griddata
import h5py
import xarray as xr
import numpy as np

# tk
from ..util.time_operations import DateConverter_Mat2Py

def ReadMatfile(p_mfile):
    'Parse .mat file to nested python dictionaries'

    def RecursiveMatExplorer(mstruct_data):
        # Recursive function to extrat mat_struct nested contents

        if isinstance(mstruct_data, mat_struct):
            # mstruct_data is a matlab structure object, go deeper
            d_rc = {}
            for fn in mstruct_data._fieldnames:
                d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
            return d_rc

        else:
            # mstruct_data is a numpy.ndarray, return value
            return mstruct_data

    # base matlab data will be in a dict
    mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
    mdata_keys = [x for x in mdata.keys() if x not in
                  ['__header__','__version__','__globals__']]

    # use recursive function
    dout = {}
    for k in mdata_keys:
        dout[k] = RecursiveMatExplorer(mdata[k])
    return dout


def ReadNakajoMats(p_mfiles):
    '''
    Read Nakajo simulated hurricanes data from .mat files folder.
    Return xarray.Dataset
    '''

    n_sim = 10
    max_tc_len = 900

    var_names = [
        'yts', 'ylon_TC' , 'ylat_TC', 'yDIR', 'ySPEED','yCPRES','del_reason'
    ]

    # find number and time length of synthetic storms
    ini_s=0
    for i in range(n_sim):
        print(str(i) + '/' + str(n_sim) )

        # generate output
        xds_out = xr.Dataset(
        {
            'yts':(('storm','time'), np.zeros((100000, max_tc_len))*np.nan),
            'ylon_TC':(('storm','time'), np.zeros((100000, max_tc_len))*np.nan),
            'ylat_TC':(('storm','time'), np.zeros((100000, max_tc_len))*np.nan),
            'yDIR':(('storm','time'), np.zeros((100000, max_tc_len))*np.nan),
            'ySPEED':(('storm','time'), np.zeros((100000, max_tc_len))*np.nan),
            'yCPRES':(('storm','time'), np.zeros((100000, max_tc_len))*np.nan),
            'del_reason':(('storm','time'), np.zeros((100000, max_tc_len))*np.nan),
        },
        coords = {
            'storm':(('storm'), np.arange(ini_s, ini_s+100000)),
        },
        )


        # read sim file
        p_matf = op.join(p_mfiles, 'YCAL{0}.mat'.format(i+1))
        d_matf = ReadMatfile(p_matf)

        # add data to output
        for vn in var_names:
            for s in range(len(d_matf[vn])):
                if isinstance(d_matf[vn][s], np.ndarray):
                    xds_out[vn][s,:len(d_matf[vn][s])] = d_matf[vn][s]

        # add sims
        if i ==0:
            xds_out_sims = xds_out.copy(deep=True)
        else:
            xds_out_sims = xr.concat([xds_out_sims, xds_out], dim = 'storm')

        # update counter
        ini_s = len(xds_out_sims.storm)


    return xds_out_sims
