#!/usr/bin/env python
# -*- coding: utf-8 -*-


# pip
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t, genextreme
from scipy.special import ndtri  # norm inv

import xarray as xr
from datetime import datetime
from matplotlib import gridspec


#bluemath path
import sys

import os
import os.path as op
#sys.path.insert(0, os.path.join(os.path.abspath(''), '..', '..', '..', '..'))


import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
from numpy.random import choice, multivariate_normal
from scipy.stats import  genextreme, gumbel_l, spearmanr, norm, weibull_min

from scipy.stats import poisson, genpareto, expon
from scipy.stats import bernoulli



# ECDF (EMPIRICAL)

from statsmodels.distributions.empirical_distribution import ECDF

def Empirical_CDF(x):
    '''
    Returns empirical cumulative probability function at x.
    '''

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    return cdf

def Empirical_ICDF(x, p):
    '''
    Returns inverse empirical cumulative probability function at p points
    '''

    # TODO: revisar que el fill_value funcione correctamente

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    # interpolate KDE CDF to get support values 
    fint = interp1d(
        cdf, x,
        fill_value=(np.nanmin(x), np.nanmax(x)),
        #fill_value=(np.min(x), np.max(x)),
        bounds_error=False
    )
    return fint(p)




def copulafit(u):
    '''
    Fit copula to data.
    Returns correlation matrix and degrees of freedom for t student
    '''

    # rhohat = None  # correlation matrix
    u[u<=0.0] = 0.000001 # confirmamos que no hay valores superiores a 1
    u[u>=1.0] = 0.999999 # confirmamos que no hay valores superiores a 1
    inv_n = ndtri(u) # Convertimos los valores uniformemente distribuidos de la variable en valores que siguen una distribución normal estándar
    rhohat = np.corrcoef(inv_n.T) # matriz de correlación de los datos transformados. Esta matriz de correlación se utiliza como estimación de la estructura de dependencia entre las variables aleatorias originales.


    return rhohat



def copularnd(rhohat, n):  # rhohat: matriz de correlacion; n: muestras que se generaran
    '''
    Random vectors from a copula
    '''
    mn = np.zeros(rhohat.shape[0]) # vector de zeros con la primera dimension de la matriz rhohat
    np_rmn = np.random.multivariate_normal(mn, rhohat, n) # Se generan n muestras aleatorias a partir de una distribución normal multivariada con media cero y matriz de covarianza rhohat
    u = norm.cdf(np_rmn) # Se transforman los valores generados a través de la función de distribución acumulativa de la distribución normal estándar, para obtener valores que sigan una distribución uniforme entre 0 y 1.


    return u



def CopulaSimulation(U_data, kernels, num_sim):
    '''
    Fill statistical space using copula simulation

    U_data: 2D nump.array, each variable in a column
    kernels: list of kernels for each column at U_data (KDE | GPareto)
    num_sim: number of simulations
    '''

    # kernel CDF dictionary (input defined for each variable)
    d_kf = {
        'ECDF' : (Empirical_CDF, Empirical_ICDF)
    }

    # check kernel input
    if any([k not in d_kf.keys() for k in kernels]):
        raise ValueError(
            'wrong kernel: {0}, use: {1}'.format(
                kernels, ' | '.join(d_kf.keys())
            )
        )

    # NORMALIZE: calculate data CDF using kernels
    U_cdf = np.zeros(U_data.shape) * np.nan
    ic = 0
    for d, k in zip(U_data.T, kernels):
        cdf, _ = d_kf[k]  # get kernel cdf
        U_cdf[:, ic] = cdf(d)
        ic += 1

    # fit data CDFs to a gaussian copula 
    rhohat = copulafit(U_cdf)

    # simulate data to fill probabilistic space
    U_cop = copularnd(rhohat, num_sim)

    # DE-NORMALIZE: calculate data ICDF
    U_sim = np.zeros(U_cop.shape) * np.nan
    ic = 0
    for d, c, k in zip(U_data.T, U_cop.T, kernels):
        _, icdf = d_kf[k]  # get kernel icdf
        U_sim[:, ic] = icdf(d, c)
        ic += 1

    return U_sim



def Copula_Hs_Tp_Dir_ss(main_path,ds,num_clusters,kernels,names,num_sim_rnd): 
    
    
    for aa in range(num_clusters):    
    
        pos=np.where(ds.bmus==aa)[0]
        print(aa)
        variables=np.column_stack((ds.Hs.values[pos], ds.Tp.values[pos], ds.Dir.values[pos], ds.ss.values[pos])) # SI METEMOS MAS VARIABLES HAY QUE CAMBIARLO

        # Quitamos nans
        mask = ~np.isnan(variables[:,0]) & ~np.isnan(variables[:,1])  & ~np.isnan(variables[:,2]) & ~np.isnan(variables[:,3])
        variables=variables[mask,:]
        # print(variables)
        # print(variables.shape)   


        # Limitador fisico
        copula = CopulaSimulation(variables, kernels, 3*num_sim_rnd)
        var_max = 2.5*np.nanmax(variables, axis=0)
        pos_copula = np.where((copula[:,0]<var_max[0]) & (copula[:,0]>=0) & (copula[:,1]<var_max[1]) & (copula[:,2]<var_max[2]) & (copula[:,3]<var_max[3]))[0]


        # Nos quedamos con la cantidad de datos que queremos
        copula = copula[pos_copula[:num_sim_rnd],:]


        fig = plt.figure(figsize=[19,5])
        gs2=gridspec.GridSpec(1,4)
        for nn in range(np.shape(variables)[1]):
            ax2=fig.add_subplot(gs2[nn])
            ax2.hist(variables[:,nn],density=True,label='Data',color='royalblue',alpha=0.7)
            ax2.hist(copula[:,nn],density=True,label='Copula',color='orchid',alpha=0.5)
            ax2.set_title(names[nn],fontsize=13)
            ax2.legend()    

        
        
        path_save=os.path.join(main_path,'Figures')
        if not os.path.exists(path_save):
            os.mkdir(path_save)

        fig.savefig(os.path.join(main_path,'Figures','copula'+str(aa)+'.png'),dpi=500)
        plt.close()


        fig, axs = plt.subplots(1, 3, figsize = [15, 5])
        axs[0].scatter(variables[:,0], variables[:,1])
        axs[0].set_xlabel('Hs')
        axs[0].set_xlabel('Tp')

        axs[1].scatter(variables[:,0], variables[:,3])
        axs[1].set_xlabel('Hs')
        axs[1].set_xlabel('ss')

        axs[2].scatter(variables[:,2], variables[:,3])
        axs[2].set_xlabel('Tp')
        axs[2].set_xlabel('ss')

        fig.savefig(os.path.join(main_path,'Figures','copula_scatter'+str(aa)+'.png'),dpi=500)
        plt.close()

        Copula_params = xr.Dataset(
            {'Hs_cop':(('num'),copula[:,0]),
            'Tp_cop':(('num'), copula[:,1]),
            'Dir_cop':(('num'), copula[:,2]),
            'ss_cop':(('num'), copula[:,3]),
            },coords = {'num':(('num'), np.arange(num_sim_rnd))})
        
        #Save
        Copula_params.to_netcdf(path=os.path.join(main_path,'Results','Copula_Parameters_'+ str(aa) +'.nc'),mode='w')


