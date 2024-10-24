import numpy as np
import xarray as xr

from ..toolkit.statistical import copula_simulation



def filter_sea_params(params_sim):

    # filter params outside boundaries
    ind_hs = np.where(params_sim[:,0]<0)[0]
    ind_tp = np.where(params_sim[:,1]<0)[0]
    ind_dir_min = np.where(params_sim[:,2]<0)[0]
    ind_dir_max = np.where(params_sim[:,2]>360)[0]
    ind_ws = np.where((params_sim[:,0]/(1.56*params_sim[:,1]**2))>0.06)[0] # wave steepness

    ind = np.unique(np.concatenate((ind_hs, ind_tp, ind_dir_min,ind_dir_max, ind_ws)))

    params_sim[ind,:] = np.nan
    params_sim = params_sim[~np.isnan(params_sim[:,0]),:]

    return params_sim


def sea_copula_sim(n_clusters, sea_states, n_data, sim_params, num_sim_rnd, kernels):
    ''' simulate for each DWT and AWT
    '''

    cont=0
    kernels_emp = ['ECDF', 'ECDF', 'ECDF']

    for aa in range(6):
        for dd in range(n_clusters):

            #sea_states_d = sea_states.where(sea_states.bmus_noTCs==dd, drop=True)
            sea_states_d = sea_states.where(sea_states.bmus==dd, drop=True)
            sea_states_d = sea_states_d.where(sea_states_d.awt==aa,drop=True)

            print('AWT ' + str(aa+1) + '. DWT ' + str(dd+1) + '. ' + str(len(sea_states_d.time)) + ' data')


            # number of historical data for each dwt and awt
            n_data[dd, aa]=len(sea_states_d.time)

            # join storm parameters for copula simulation
            params = np.column_stack(
                (sea_states_d.hs, sea_states_d.tp, sea_states_d.dpm)
            )

            # statistical simulate seas using copulas
            if len(sea_states_d.time)==0:
                params_sim = np.empty((1,np.shape(params)[1]))*np.nan
                print('empty')
                cont = cont+1
                continue

            elif len(sea_states_d.time)<=5:
                params_sim = np.tile(params,(num_sim_rnd*10,1)) # simular de mas
                print('repeat')

            elif len(sea_states_d.time)<50:

                params_sim = copula_simulation(params, kernels_emp, num_sim_rnd*10) # simular de mas
                print('ecdf')

            else:
                params_sim = copula_simulation(params, kernels, num_sim_rnd*10) # simular de mas
                print('kde')

            # filter params outside boundaries
            params_sim = filter_sea_params(params_sim)

            # keep needed variables
            params_sim = params_sim[:num_sim_rnd,:]

            # store simulated sea state - parameters
            sim_params['hs'][aa,dd,:] = params_sim[:,0]
            sim_params['tp'][aa,dd,:] = params_sim[:,1]
            sim_params['dpm'][aa,dd,:] = params_sim[:,2]

            cont = cont+1

    return n_data, sim_params


def normalize_vars(snakes_params):
    ''' Normalize variables
    '''
    params_norm = np.column_stack((snakes_params.Hs, snakes_params.Tp, snakes_params.Dir, snakes_params.Duration,
         snakes_params.Tau, snakes_params.sh1, snakes_params.sf1, snakes_params.H_ini ))

    params_min = np.min(params_norm, axis=0)
    params_max = np.max(params_norm, axis=0)

    params_norm = (params_norm - params_min)/(params_max-params_min)

    # normalize dir
    params_norm[:,2] = snakes_params.Dir.values/360

    return params_norm


def find_k_nearest(snakes_params, params_norm, k):
    ''' Select closest swells to each swell
    '''

    # define output
    snakes_params_nearest = xr.Dataset(
        {
            'Hs':(('nearest','time'), np.zeros((k, len(snakes_params.time)))*np.nan),
            'Tp':(('nearest','time'), np.zeros((k, len(snakes_params.time)))*np.nan),
            'Dir':(('nearest','time'), np.zeros((k, len(snakes_params.time)))*np.nan),
            'Duration':(('nearest','time'), np.zeros((k, len(snakes_params.time)))*np.nan),
            'Tau':(('nearest','time'), np.zeros((k, len(snakes_params.time)))*np.nan),
            'sh1':(('nearest','time'), np.zeros((k, len(snakes_params.time)))*np.nan),
            'sf1':(('nearest','time'), np.zeros((k, len(snakes_params.time)))*np.nan),
            'H_ini':(('nearest','time'), np.zeros((k, len(snakes_params.time)))*np.nan),
            'bmus':(('time'), snakes_params.bmus),
        },
        coords = {
            'time':(('time'), snakes_params.time),
            'nearest':(('nearest'), np.arange(k)),
        },)


    for p in range(0,len(params_norm)):
        params_norm_p = params_norm[p,:]

        # euclidian distance to all points
        dif = (params_norm - params_norm_p)**2
        dif_dir = abs(params_norm[:,2] - params_norm_p[2])
        dif_dir = np.where(dif_dir<=.5, dif_dir, abs(dif_dir-1)) #dif_dir = np.where(dif_dir<=180, dif_dir, abs(dif_dir-360))
        dif[:,2] = dif_dir**2
        params_norm_p_dist = np.sqrt(np.sum(dif,axis=1))

        # select nearest
        k_nearest = np.argsort(params_norm_p_dist)[:k]

        # save nearest
        snakes_params_nearest['Hs'][:,p] = snakes_params.Hs.isel(time=k_nearest).values
        snakes_params_nearest['Tp'][:,p] = snakes_params.Tp.isel(time=k_nearest).values
        snakes_params_nearest['Dir'][:,p] = snakes_params.Dir.isel(time=k_nearest).values
        snakes_params_nearest['Duration'][:,p] = snakes_params.Duration.isel(time=k_nearest).values
        snakes_params_nearest['Tau'][:,p] = snakes_params.Tau.isel(time=k_nearest).values
        snakes_params_nearest['sh1'][:,p] = snakes_params.sh1.isel(time=k_nearest).values
        snakes_params_nearest['sf1'][:,p] = snakes_params.sf1.isel(time=k_nearest).values
        snakes_params_nearest['H_ini'][:,p] = snakes_params.H_ini.isel(time=k_nearest).values


    return snakes_params_nearest


def snakes_k_nearest_sim(snakes_params, n_data, sim_params, k, dir_ini, dir_end, n_clusters, num_sim_rnd):
    '''
     simulate for each DWT and direction
    '''
    cont_dir=0
    for dir1, dir2 in zip(dir_ini, dir_end):

        # data for selecting nearest (all DWTs, this directional sector)
        snakes_params_dir = snakes_params.where((snakes_params.Dir>=dir1) & (snakes_params.Dir<dir2), drop=True)

        # normalize variables
        params_norm = normalize_vars(snakes_params_dir)

        # select k-nearest
        snakes_params_nearest = find_k_nearest(snakes_params_dir, params_norm, k)

        # simulate for each DWT
        for dd in range(0,n_clusters):

            # data for this DWT and directional sector
            snakes_params_dir_wt = snakes_params_dir.where(snakes_params_dir.bmus==dd, drop=True)

            print('DWT ' + str(dd+1) + ' dir ' + str(dir1) + '-' + str(dir2) + '. ' + str(len(snakes_params_dir_wt.time)) + ' data')


            # number of historical data for each dwt and dir
            n_data[dd, cont_dir]=len(snakes_params_dir_wt.time)

            # simulate snakes
            if len(snakes_params_dir_wt.time)==0:
                continue

            #elif len(snakes_params_wt_dir.time)<k+1:
            #    print('repeat')
            #    params = np.column_stack((snakes_params_wt_dir.Hs, snakes_params_wt_dir.Tp, snakes_params_wt_dir.Dir, snakes_params_wt_dir.Duration,
            #                             snakes_params_wt_dir.Tau, snakes_params_wt_dir.sh1, snakes_params_wt_dir.sf1, snakes_params_wt_dir.H_ini ))
            #    params_sim = np.tile(params,(num_sim_rnd,1))

            else:

                # simulate
                params_sim = np.empty((num_sim_rnd*len(snakes_params_dir_wt.time),len(list(sim_params.keys()))))*np.nan
                cont_t_ini = 0
                cont_t_end = len(snakes_params_dir_wt.time)

                while True:

                    snakes_params_nearest_dwt = snakes_params_nearest.where(snakes_params_nearest.bmus==dd, drop=True)

                    # random weights to each near point
                    probs = np.random.rand(len(snakes_params_nearest_dwt.nearest), len(snakes_params_nearest_dwt.time))
                    probs = probs/np.sum(probs,axis=0)

                    if cont_t_ini>num_sim_rnd:
                        break
                    else:
                        # weighted mean from near points
                        params_sim[cont_t_ini:cont_t_end,0] = np.sum(snakes_params_nearest_dwt.Hs.values*probs, axis=0)
                        params_sim[cont_t_ini:cont_t_end,1] = np.sum(snakes_params_nearest_dwt.Tp.values*probs, axis=0)
                        params_sim[cont_t_ini:cont_t_end,2] = np.sum(snakes_params_nearest_dwt.Dir.values*probs, axis=0)
                        params_sim[cont_t_ini:cont_t_end,3] = np.sum(snakes_params_nearest_dwt.Duration.values*probs, axis=0)
                        params_sim[cont_t_ini:cont_t_end,4] = np.sum(snakes_params_nearest_dwt.Tau.values*probs, axis=0)
                        params_sim[cont_t_ini:cont_t_end,5] = np.sum(snakes_params_nearest_dwt.sh1.values*probs, axis=0)
                        params_sim[cont_t_ini:cont_t_end,6] = np.sum(snakes_params_nearest_dwt.sf1.values*probs, axis=0)
                        params_sim[cont_t_ini:cont_t_end,7] = np.sum(snakes_params_nearest_dwt.H_ini.values*probs, axis=0)

                        cont_t_ini = cont_t_end
                        cont_t_end = cont_t_end+len(snakes_params_nearest_dwt.time)


            # store simulated snake - parameters
            sim_params['Hs'][dd,cont_dir,:] = params_sim[:num_sim_rnd,0]
            sim_params['Tp'][dd,cont_dir,:] = params_sim[:num_sim_rnd,1]
            sim_params['Dir'][dd,cont_dir,:] = params_sim[:num_sim_rnd,2]
            sim_params['Duration'][dd,cont_dir,:] = params_sim[:num_sim_rnd,3]
            sim_params['Tau'][dd,cont_dir,:] = params_sim[:num_sim_rnd,4]
            sim_params['sh1'][dd,cont_dir,:] = params_sim[:num_sim_rnd,5]
            sim_params['sf1'][dd,cont_dir,:] = params_sim[:num_sim_rnd,6]
            sim_params['H_ini'][dd,cont_dir,:] = params_sim[:num_sim_rnd,7]

        cont_dir=cont_dir+1

        print()

    return n_data, sim_params
