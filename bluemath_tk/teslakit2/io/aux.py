#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common 
import os
import os.path as op
import json

# pip
import netCDF4 as nc

# teslakit
from .aux_nc import StoreBugXdset
from ..__init__ import __version__, __author__


def clean_files(l_files):
    'remove files at list'
    for f in l_files:
        if op.isfile(f): os.remove(f)

def fill_metadata(xds, set_source=False):
    '''
    for each variable in xarray.Dataset xds, attributes will be set
    using resources/variables_attrs.json

    set_source - True for adding package source and institution metadata
    '''

    # read attributes dictionary
    p_resources = op.dirname(op.realpath(__file__))
    p_vats = op.join(p_resources, 'variables_attrs.json')
    with open(p_vats) as jf:
        d_vats = json.load(jf)

    # update dataset variables (names, units, descriptions)
    for vn in xds.variables:
        if vn.lower() in d_vats.keys():
           xds[vn].attrs = d_vats[vn.lower()]

    # set global attributes (source, institution)
    if set_source:
        xds.attrs['source'] = 'teslakit_v{0}'.format(__version__)
        #xds.attrs['institution'] = '{0}'.format(__author__)

    return xds

def save_nc(xds, p_save, safe_time=False):
    '''
    (SECURE) exports xarray.Dataset to netcdf file format.

     - fills dataset with teslakit variables and source metadata
     - avoids overwritting problems
     - set safe_time=True if dataset contains dates beyond 2262
    '''

    # TODO: save con netCDF4 y no xarray

    # add teslakit metadata to xarray.Dataset
    xds = fill_metadata(xds, set_source=True)

    # remove previous file to avoid problems
    clean_files([p_save])

    # export .nc
    # TODO: auto para date_end > 2260
    if safe_time:
        StoreBugXdset(xds, p_save)  # time dimension safe
    else:
        xds.to_netcdf(p_save, 'w')

