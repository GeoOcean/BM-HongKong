
import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

import sys
from ..util.time_operations import date2datenum, generate_datetimes, npdt64todatetime


def Normalize_inputs(spec_part, gamma_lim):

    #We removed the sea partition (0) from the analysis as the bahaviour is totally different
    hs=spec_part.hs.values
    tm02=spec_part.tm02.values
    tp=spec_part.tp.values
    dirs=spec_part.dpm.values
    dspr=spec_part.dspr.values
    hs[np.where(hs==0)]=np.nan
    a = 1.411
    b = -0.07972
    gamma = np.exp((np.log(tp/(a*tm02)))/b)

    # WE NORMALIZE THE VARIABLES
#    hs=(hs-np.nanmin(hs))/(np.nanmax(hs)-np.nanmin(hs)); 
    hs=(hs-np.nanpercentile(hs,0.1))/(np.nanpercentile(hs,99.9)-np.nanpercentile(hs,0.1));
    tp=(tp-np.nanmin(tp))/(np.nanmax(tp)-np.nanmin(tp))
    dspr[np.where(dspr>15)[0],np.where(dspr>15)[1]]=15; dspr=(dspr-np.nanmin(dspr))/(np.nanmax(dspr)-np.nanmin(dspr))
    
    gamma_lim=gamma_lim #!!!!
    gamma[np.where(gamma<1)[0],np.where(gamma<1)[1]] = 1
    gamma[np.where(gamma>gamma_lim)[0],np.where(gamma>gamma_lim)[1]] = gamma_lim
    gamma=np.log(gamma)
    gamma=(gamma-np.nanmin(gamma))/(np.nanmax(gamma)-np.nanmin(gamma))
    dirs=((dirs*np.pi)/180)/np.pi;
    
    spec_norm = xr.Dataset(
        {
            'hs': (['part','time'], hs),
            'tp': (['part','time'], tp),
            'dpm': (['part','time'], dirs),
            'dspr': (['part','time'], dspr),
            'gamma': (['part','time'], gamma),
        },
        coords={
            'time': spec_part.time.values,
            'part': spec_part.part.values,
        },
    )
    
      
    return spec_norm


def Isolate_snakes(spec_norm,coefs=[1,1,1,0,0],er=0.08,MaxPerc_t=0.03, MaxPerc_tp=0.25,MaxPerc_hs=0.2,max_jump=15,n_hours_min=1):
    
    '''
    This function isolates individual swell systems
    
    These are the different coefficients:
       
    - spec_norm:   Normalized partition parameters: hs, tp, dir, dspr, gamma
    - coefs:       Relative weight of each variables Hs, Tp, Dir, Dspr, Gamma
    - er:          Error for each variable (%)This is used to compute the maximum distance, not individually
    - MaxPerc_t:   Percentage period can increase (generally only decreasing)
    - MaxPerc_tp:  Percentage tp can decrease
    - MaxPerc_hs:  Percentage hs can change (increase and decrease) 
    - max_jump:    Maximum number of positions to look for the continuation of a swell train
    - n_hours_min:    Minimum snake length

    '''

    # TODO: aunque n_hours_min entra en la función, no se tiene en cuenta para obtener las snakes


    hs=np.copy(spec_norm.hs.values); tp=np.copy(spec_norm.tp.values); dirs=np.copy(spec_norm.dpm.values);
    dspr=np.copy(spec_norm.dspr); gamma=np.copy(spec_norm.gamma.values);
    
    coef_hs=coefs[0]; coef_tp=coefs[1]; coef_dir=coefs[2]; coef_dspr=coefs[3]; coef_gamma=coefs[4];
    
    MaxDist=np.sqrt(coef_hs*(er**2)+coef_tp*(er**2)+coef_dir*(er**2)+coef_dspr*(er**2)+coef_gamma*(er**2))  # Distance to consider continuity in the snake (0-sqrt(5))
    snakes=np.full([np.shape(hs)[0],np.shape(hs)[1]], np.nan)
    
    namecont=1; #Nombre de las culebras
    
    while len(np.where(~np.isnan(hs[:,:-max_jump]))[0])>=1:
        s=np.transpose(np.array(np.where(~np.isnan(hs[:,:-max_jump]))))
        time1=np.min(s[:,1]) #Minimo tiempo en el que queda valores sin nan
        parts=np.where(~np.isnan(hs[:,time1]))[0]
        parts=np.flip(parts) #Comenzamos con las de menor energia para pillar inicios
        for p in parts: 
            time=time1;
            if time<=(np.shape(hs)[1]-max_jump): 
                snakes[p,time]=namecont;
                cont_time=1
                while time<=np.shape(hs)[1]-max_jump and cont_time<max_jump:
                    dists=np.full([np.shape(hs)[0],1], np.nan)
                    perc=np.full([np.shape(hs)[0],1], np.nan)
                    perch=np.full([np.shape(hs)[0],1], np.nan)
                    perct=np.full([np.shape(hs)[0],1], np.nan)
                    for q in np.where(~np.isnan(hs[:,time+cont_time]))[0]:
                        dists[q]=np.sqrt(coef_hs*((hs[p,time]-hs[q,time+cont_time])**2) + coef_tp*((tp[p,time]-tp[q,time+cont_time])**2)  + coef_dir*((np.minimum(np.abs(dirs[p,time]-
                                         dirs[q,time+cont_time]),2-np.abs(dirs[p,time]-dirs[q,time+cont_time])))**2) 
                                         + coef_dspr*((dspr[p,time]-dspr[q,time+cont_time])**2) + coef_gamma*((gamma[p,time]-gamma[q,time+cont_time])**2))
                        perc[q]=(tp[q,time+cont_time]/tp[p,time])-1
                        perch[q]=np.abs((hs[q,time+cont_time]/hs[p,time])-1)
                        perct[q]=np.abs((tp[q,time+cont_time]/tp[p,time])-1)


                    if len(np.where(~np.isnan(dists))[0])<1:
                        mperc=np.nan; mperch=np.nan; mperct=np.nan;
                    else:
                        mdist=np.nanmin(dists); pos=np.where(dists==mdist)
                        mperc=perc[pos[0]]; #Maximum percentual change
                        mperch=perch[pos[0]]; #Maximum percentual change
                        mperct=perct[pos[0]]; #Maximum percentual change
                        
                    if hs[p,time]<=0.2 and mperc<MaxPerc_t  and mperct<MaxPerc_tp and mdist<MaxDist:
                        hs[p,time]=np.nan; tp[p,time]=np.nan; dirs[p,time]=np.nan; dspr[p,time]=np.nan; gamma[p,time]=np.nan;
                        p=pos[0]
                        time=time+cont_time
                        snakes[p,time]=namecont
                        cont_time=1
                    elif mperc<MaxPerc_t and mperch<MaxPerc_hs and mperct<MaxPerc_tp and mdist<MaxDist:
                        hs[p,time]=np.nan; tp[p,time]=np.nan; dirs[p,time]=np.nan; dspr[p,time]=np.nan; gamma[p,time]=np.nan;
                        p=pos[0]
                        time=time+cont_time
                        snakes[p,time]=namecont
                        cont_time=1
                    else:
                        cont_time=cont_time+1
                        
                    if cont_time==max_jump:
                        hs[p,time]=np.nan; tp[p,time]=np.nan; dirs[p,time]=np.nan; dspr[p,time]=np.nan; gamma[p,time]=np.nan;
                                                            
                
                if len(np.where(snakes==namecont)[1])>n_hours_min: #Min number of points to consider a snake
                    ord=np.argsort(np.where(snakes==namecont)[1])
                    tt=np.where(snakes==namecont)[1][ord]
                    value=np.where(snakes==namecont)[0][ord]

                namecont=namecont+1;
    
    return snakes, namecont


def parameterize_Snakes(path,spec_part, snakes, posi, func_hs, func_tp, func_dir, func_dspr, func_gamma):


    a = 1.411; b = -0.07972
    times_p = np.datetime64('1000-01-01T01:00:00.000000000');
    cont = 1
    duration=[]; tau=[]; h_s=[];  h_initial=[]; tm_s=[]; tp_s=[];  fp_s=[]; d_s=[]; g_s=[]; dp_s=[];
    sh1=[]; sf1=[]; sd1=[]; sg1=[]; sdp1=[];

    for i in range(len(posi)-1):
        d = np.int(posi[i])
        orde = np.argsort(np.where(snakes==d)[1])
        time = np.where(snakes==d)[1][orde]
        part = np.where(snakes==d)[0][orde]

        h = spec_part.hs.values[part, time]
        t = spec_part.tp.values[part, time]
        tm = spec_part.tm02.values[part, time]
        d = spec_part.dpm.values[part, time]
        dp = spec_part.dspr.values[part, time]
        g = np.exp((np.log(t/(a*tm)))/b)
        pm = np.argmax(h)

        time_real = spec_part.time[time]
        timeplot = time-time[0]

        if len(h)<5: # snake with enough length but not enough data (too many or too large jumps)
            continue

        # continuous hourly time without jumps (for duration and tau)
        t_h = pd.date_range(start=time_real.values[0], end=time_real.values[-1], freq='H')
        pm_h = np.where(t_h.values==time_real.isel(time=pm).values)

        if len(np.unique(t))>=2:

            if len(t[pm:])<3: # Para poder calcular la pendiente
                pm=pm-3


            #-----------------------
            # parameterize

            # sh1
            popth, pcov = curve_fit(func_hs, timeplot[pm:], h[pm:],method= 'lm',maxfev=20000)

            # sf1
            poptt, pcov = curve_fit(func_tp, timeplot, 1/t, method= 'lm',maxfev=20000)

            # sd1
            if np.nanmin(d)<30 and np.nanmax(d)>330 and d[pm]<30:
                d[np.where(d>330)[0]] = d[np.where(d>330)[0]]-360
            if np.nanmin(d)<30 and np.nanmax(d)>330 and d[pm]>330:
                d[np.where(d<30)[0]] = d[np.where(d<30)[0]]+360
            poptd, pcov = curve_fit(func_dir, timeplot, d, method= 'lm',maxfev=20000)

            # sdp1
            poptdp, pcov = curve_fit(func_dspr, timeplot, dp,method= 'lm',maxfev=20000)

            # sg1
            poptg, pcov = curve_fit(func_gamma, timeplot, g,method= 'lm',maxfev=20000)


            # filter: Hs slope must be negative and Fp slope must be positive
            if popth[1]<0 and poptt[1]>0:

                #-----------------------
                # save individual snakes
                wvars = xr.Dataset({'Hs': (['time'],h), 'Tp': (['time'],t) ,'Fp': (['time'],1/t) , 'Tm': (['time'],tm),
                                    'Dir': (['time'],d), 'Dp': (['time'],dp), 'Gamma': (['time'],g)},
                                   coords={'time': time_real})

                ty = time_real[0].dt.year.values.astype('int')
                wvars.to_netcdf(path=os.path.join(path,'Snake_' + str(cont) + '_' + str(ty) +'.nc'))
                cont = cont+1


                #-----------------------
                # save parameters
                tau = np.append(tau, pm_h)
                duration = np.append(duration,len(t_h))
                times_p = np.append(times_p, time_real[0]) # t de inicio de la snake
                h_s = np.append(h_s, h[pm])
                sh1 = np.append(sh1, popth[1])
                h_initial = np.append(h_initial, h[0])

                tm_s = np.append(tm_s, tm[0])
                tp_s = np.append(tp_s, t[0])
                fp_s = np.append(fp_s,1/(t[0]))
                sf1 = np.append(sf1, poptt[1])

                d_s = np.append(d_s, d[pm])
                sd1 = np.append(sd1, poptd[1])
                dp_s = np.append(dp_s, dp[pm])
                sdp1 = np.append(sdp1, poptdp[1])
                g_s = np.append(g_s, g[pm])
                sg1 = np.append(sg1, poptg[1])


    #-----------------------
    # save snake parameters
    times_p = times_p[1:]
    parameters = xr.Dataset({'Tau': (['time'],tau), 'Duration': (['time'],duration), 'Hs': (['time'],h_s),
                             'H_ini': (['time'],h_initial), 'sh1': (['time'],sh1), 'Tp': (['time'],tp_s), 'Fp': (['time'],fp_s),
                             'Tm': (['time'],tm_s),  'sf1': (['time'],sf1), 'Dir': (['time'],d_s), 'sd1': (['time'],sd1),
                             'Dspr': (['time'],dp_s), 'sdp1': (['time'],sdp1) , 'Gamma': (['time'],g_s),  'sg1': (['time'],sg1)},
                            coords={'time': times_p})

    return parameters



def add_bmus(data, KMA):

    bmus = np.zeros(len(data.time.values))*np.nan
    bmus_noTCs = np.zeros(len(data.time.values))*np.nan

    t_day=[]

    for ind_t, t in enumerate(data.time.values):
        t = npdt64todatetime(t)
        t = t.replace(hour=0)
        bmus[ind_t] = KMA.sorted_bmus_storms.where(KMA.time==np.datetime64(t), drop=True).values
        bmus_noTCs[ind_t] = KMA.sorted_bmus.where(KMA.time==np.datetime64(t), drop=True).values

        t_day.append(np.datetime64(t))

    data['bmus'] = (('time'), bmus)
    data['bmus_noTCs'] = (('time'), bmus_noTCs)
    data['time_d'] = (('time'), t_day)

    return data


def group_by_dwt_dir_t(snakes, KMA, n_clusters=42,amp_dir=15,amp_t=5,t_max=30,):

    # bins dir
    dir_ini = np.arange(0,360,amp_dir)
    dir_end = np.arange(amp_dir,360+amp_dir,amp_dir)

    # bins period
    t_ini = np.arange(0, t_max,amp_t)
    t_end = np.arange(amp_t,t_max+amp_t,amp_t)

    # initialize variables
    n_dwts = np.zeros((1, n_clusters))*np.nan
    n_snakes = np.zeros((n_clusters, len(t_ini), len(dir_ini)))*np.nan
    hs_snakes = np.zeros((n_clusters, len(t_ini), len(dir_ini)))*np.nan
    tp_snakes = np.zeros((n_clusters, len(t_ini), len(dir_ini)))*np.nan

    # loop for each WT
    for b in range(0, n_clusters):

        if n_clusters==42:
            # number of days in each DWT
            dwt = KMA.where(KMA.sorted_bmus_storms==b,drop=True)
            n_dwts[0, b] = len(dwt.time)

            # snakes for each DWT
            snake_dwt = snakes.where(snakes.bmus==b, drop=True)

        elif n_clusters==36:
            # number of days in each DWT
            dwt = KMA.where(KMA.sorted_bmus==b,drop=True)
            n_dwts[0, b] = len(dwt.time)

            # snakes for each DWT
            snake_dwt = snakes.where(snakes.bmus_noTCs==b, drop=True)


        # loop for each direction
        cont_dir=0
        for dir1, dir2 in zip(dir_ini, dir_end):

            snake_dwt_dir = snake_dwt.where((snake_dwt.Dir>=dir1) & (snake_dwt.Dir<dir2), drop=True)

            # loop for each period
            cont_t=0
            for t1, t2 in zip(t_ini, t_end):

                snake_dwt_dir_t = snake_dwt_dir.where((snake_dwt_dir.Tp>=t1) & (snake_dwt_dir.Tp<t2), drop=True)

                # number of snakes in each DWT and for each direction and period
                n_snakes[b, cont_t, cont_dir] = len(snake_dwt_dir_t.time)

                # mean hs & tp of snakes in each DWT and for each direction  and period
                hs_snakes[b, cont_t, cont_dir] = np.mean(snake_dwt_dir_t.Hs.values)
                tp_snakes[b, cont_t, cont_dir] = np.mean(snake_dwt_dir_t.Tp.values)

                cont_t=cont_t+1

            cont_dir=cont_dir+1

    return n_dwts[0], n_snakes, hs_snakes


def group_by_day_dir(snakes, n_clusters=42, amp_dir=15):

    # keep needed variables
    snakes = snakes[['Dir','bmus','bmus_noTCs','time_d']]

    # generate time in days
    t = pd.date_range(start=snakes.time.values[0], end=snakes.time.values[-1], freq='D', normalize=True)
    #t = t[:20]

    # bins dir
    dir_ini = np.arange(0,360,amp_dir)
    dir_end = np.arange(amp_dir,360+amp_dir,amp_dir)

    # initialize output variable
    n_snakes = np.zeros((len(t), len(dir_ini)))*np.nan
    dwt = np.zeros((len(t)))*np.nan

    # loop for each day
    for ind_d, d in enumerate(t):

        snakes_d = snakes.where(snakes.time_d==d, drop=True)

        # loop for each direction
        cont_dir=0
        for dir1,dir2 in zip(dir_ini, dir_end):

            snake_dir = snakes_d.where((snakes_d.Dir>=dir1) & (snakes_d.Dir<dir2), drop=True)
            n_snakes[ind_d,cont_dir] = len(snake_dir.time)

            cont_dir=cont_dir+1

        # save dwt
        if len(snakes_d.time.values)>=1:
            if n_clusters==42:
                dwt[ind_d] = snakes_d.bmus.values[0]
            elif n_clusters==36:
                dwt[ind_d] = snakes_d.bmus_noTCs.values[0]


    n_swells = xr.Dataset(
        {'n_swells': (('time', 'n_dirs'), n_snakes),
         'bmus': (('time',), dwt)},
        coords = {'time': t,
                 'n_dirs': dir_ini}
        )

    return n_swells




def reconstruct_snakes(max_num_snakes, snake_params_sim_y, type):

    # generate output time array
    time_base = snake_params_sim_y.time.values[:]
    t0, t1 = date2datenum(time_base[0]), date2datenum(time_base[-1])
    time_h = generate_datetimes(t0, t1, dtype='datetime64[h]')
    time_h = time_h[:-1]

    # generate output variables
    hs_swell=np.full([max_num_snakes*len(snake_params_sim_y.time),len(time_h)],np.NaN)
    tp_swell=np.full([max_num_snakes*len(snake_params_sim_y.time),len(time_h)],np.NaN)
    dir_swell=np.full([max_num_snakes*len(snake_params_sim_y.time),len(time_h)],np.NaN)

    cont_snake = 0
    cont_day = 0
    for t in snake_params_sim_y.time.values:

        # select data for each day
        snake_params_sim_day = snake_params_sim_y.sel(time=t)
        ind_noNaN = np.where(~np.isnan(snake_params_sim_day.Hs.values))[0]

        if ind_noNaN.size==0: # there are no snakes that day
            cont_day = cont_day + 24
            continue
        else:
            snake_params_sim_day = snake_params_sim_day.isel(n_snake = ind_noNaN)


        # Hs
        yini = snake_params_sim_day.H_ini.values
        y = snake_params_sim_day.Hs.values
        b = snake_params_sim_day.sh1.values
        x = snake_params_sim_day.Tau.values
        x1 = snake_params_sim_day.Duration.values
        a = y - b*x
        y1 = a + b*x1
        Hs = np.array([yini, y, y1]).T # Hs at ini, at Hs max and at the end of the snake

        # Tp
        y = 1/snake_params_sim_day.Tp.values # frecuency
        bt = snake_params_sim_day.sf1.values
        at = y - bt*0
        y2 = at + bt*x # ini
        y1 = at + bt*x1 # end
        Fp = np.array([y, y2, y1]).T

        # Dir
        Dir = snake_params_sim_day.Dir.values
        Dir = np.array([Dir, Dir, Dir]).T

        # hours
        t = np.array([np.full(len(y),0), snake_params_sim_day.Tau.values, snake_params_sim_day.Duration.values]).T

        # interpolate each snake to hourly
        for a in range(len(y)):

            # hours
            t_a = [int(round(i)) for i in t[a,:]] # Tau and duration should be rounded hours
            x_h = np.arange(t_a[0], t_a[-1],1)

            # Hs
            fh = interp1d(t_a, Hs[a,:])
            Hs_h = fh(x_h)

            # Tp
            ft = interp1d(t_a, Fp[a,:])
            Tp_h = 1/ft(x_h)

            # Dir
            fd = interp1d(t_a, Dir[a,:])
            Dir_h = fd(x_h)

            # time positon
            if type=='his':
                h_rnd = int(snake_params_sim_day.t_ini.values[a]) # snake at the time it occured
            elif type=='sim':
                h_rnd = np.random.randint(0,24) # snake at random initial time
            ind = h_rnd + x_h + cont_day


            # snake goes into next year: cut it
            if ind[-1]>=len(time_h):
                ind = ind[ind<len(time_h)]

            # add
            hs_swell[cont_snake, ind] = Hs_h[:len(ind)]
            tp_swell[cont_snake, ind] = Tp_h[:len(ind)]
            dir_swell[cont_snake, ind] = Dir_h[:len(ind)]

            cont_snake=cont_snake+1

        cont_day = cont_day + 24


    # delete not needed rows
    for r in range(len(hs_swell)):
        data = np.where(~np.isnan(hs_swell[r,:]))[0]
        if len(data)==0:
            break

    hs_swell = hs_swell[:r-1,:]
    tp_swell = tp_swell[:r-1,:]
    dir_swell = dir_swell[:r-1,:]


    # al reconstruir la duración de la snake, la Hs puede quedar negativa --> poner NaNs
    tp_swell =  np.where(hs_swell<0, np.nan, tp_swell)
    dir_swell = np.where(hs_swell<0, np.nan, dir_swell)
    hs_swell =  np.where(hs_swell<0, np.nan, hs_swell)

    # TODO: hace falta limitar el periodo??


    return time_h, hs_swell, tp_swell, dir_swell



def Aggregate_Snakes(hs_swell, tp_swell, dir_swell, times, a_tp='quadratic'):
    '''
    Aggregate Hs, Tp and Dir from snakes

    a_tp = 'quadratic' / 'max_energy', Tp aggregation formulae

    returns Hs, Tp, Dir (numpy.array)
    '''

    # Hs from families
    HS = np.sqrt(np.nansum(np.power(hs_swell,2), axis=0))

    # nan positions
    ix_nan_data = np.where(HS==0)

    # Tp
    if a_tp == 'quadratic':

        # TP from families 
        tmp1 = np.power(hs_swell,2)
        tmp2 = np.divide(np.power(hs_swell,2), np.power(tp_swell,2))
        TP = np.sqrt(np.nansum(tmp1, axis=0) / np.nansum(tmp2, axis=0))

    elif a_tp == 'max_energy':

        # Hs maximun position
        vv_Hs_nanzero = hs_swell.copy()
        vv_Hs_nanzero[np.isnan(hs_swell)] = 0
        p_max_hs = np.nanargmax(vv_Hs_nanzero, axis=0)

        # Tp from families (Hs max pos)
        TP = np.array([r[i] for r,i in zip(tp_swell.T, p_max_hs)])

    else:
        # TODO: make it fail
        pass


    # Dir from families
    #tmp3 = np.arctan2(
    #    np.nansum(np.power(hs_swell,2) * tp_swell * np.sin(dir_swell * np.pi/180), axis=0),
    #    np.nansum(np.power(hs_swell,2) * tp_swell * np.cos(dir_swell * np.pi/180), axis=0)
    #)
    #tmp3[tmp3<0] = tmp3[tmp3<0] + 2*np.pi
    #DIR = tmp3 * 180/np.pi

    # Dir from families (Hs max pos)
    DIR = np.array([r[i] for r,i in zip(dir_swell.T, p_max_hs)])

    # clear nans
    HS[ix_nan_data] = np.nan
    TP[ix_nan_data] = np.nan
    DIR[ix_nan_data] = np.nan


    # return xarray.Dataset
    xds_AGGR = xr.Dataset(
        {
            'Hs_swell': (('time',), HS),
            'Tp_swell': (('time',), TP),
            'Dir_swell': (('time',), DIR),
        },
        coords = {
            'time': times,  # get time from input
        }
    )

    return xds_AGGR
