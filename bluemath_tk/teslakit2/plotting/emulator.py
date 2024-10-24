import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .superpoint_partitions import axplot_spectrum

def plot_his_sim_waves_hist(waves_his, waves_sim, title=''):
    # Compare historical and simulated swells (all DWTs)

    _faspect = 1.618
    _fsize = 9.8

    n_rows = 3
    n_cols = 3

    vars_plot = list(waves_sim.keys())

    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3) #, wspace=0.0, hspace=0.0)
    gr, gc = 0, 0
    for var in vars_plot:
        v_h = waves_his[var]
        v_s = waves_sim[var].values.flatten()

        # remove nans
        v_h = v_h[~np.isnan(v_h)]
        v_s = v_s[~np.isnan(v_s)]


        ax = plt.subplot(gs[gr, gc])


        lim_min = np.minimum(np.nanmin(v_h), np.nanmin(v_s))
        lim_max = np.maximum(np.nanmax(v_h), np.nanmax(v_s))

        bins = np.linspace(lim_min, lim_max, 40)

        ax.hist(v_h, bins=bins, weights=np.ones(len(v_h)) / len(v_h),
                alpha=0.9, color='white', ec='k', label = 'Historical')

        ax.hist(v_s, bins=bins, weights=np.ones(len(v_s)) / len(v_s),
                alpha=0.7, color='skyblue', ec='k', label = 'Simulation')

        ax.legend(prop={'size':8})
        ax.set_title(var)

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    plt.suptitle(title);

def Plot_BulkParameters_Validation(bulk_sim,bulk_his,variables, nsims,sim,opt):

    fig = plt.figure(figsize=[15.6,11.4])
    gs1=gridspec.GridSpec(3,1)
    ax1=fig.add_subplot(gs1[0])
    ax2=fig.add_subplot(gs1[1],sharex=ax1)
    ax3=fig.add_subplot(gs1[2],sharex=ax1)

    #---------------------
    # plot time series
    #ax1.fill_between( bulk_sim.time,np.nanmin(bulk_sim.Hs.values,axis=0), np.nanmax(bulk_sim.Hs.values,axis=0), color='powderblue',alpha=0.7, label = 'Sim (Max-Min)')
    ax1.plot(bulk_sim.time, bulk_sim[variables[0]],'-',color='mediumpurple',linewidth=0.5,label='Sim: ' + str(sim))
    ax1.plot(bulk_his.time,bulk_his[variables[0]],'-',color='lightcoral',linewidth=0.3,label='Hindcast')
    # ax1.legend()
    ax1.set_xlim(np.nanmin(bulk_sim.time),np.nanmax(bulk_sim.time))
    ax1.legend(ncol=3)

    #ax2.fill_between( bulk_sim.time,np.nanmin(bulk_sim.Tm.values,axis=0), np.nanmax(bulk_sim.Tm.values,axis=0), color='powderblue',alpha=0.7, label = 'Sim (Max-Min)')
    ax2.plot(bulk_sim.time,(bulk_sim[variables[1]]),'-',color='mediumpurple',linewidth=0.5,label='Sim: ' + str(sim))
    ax2.plot(bulk_his.time,bulk_his[variables[1]],'-',color='lightcoral',linewidth=0.3,label='Hindcast')
    # ax2.legend()

#    for a in range(nsims):
#        ax3.plot(bulk_sim.time,bulk_sim.Dir[a,:],'.',color='powderblue',markersize=0.3,label='Sim')
    ax3.plot(bulk_sim.time,bulk_sim[variables[2]],'.',color='mediumpurple',markersize=0.3,label='Sim ' + str(sim))
    ax3.plot(bulk_his.time,bulk_his[variables[2]],'.',color='lightcoral',markersize=0.3,label='Hindcast')
    # ax3.legend()

    #ax1.legend()
    ax1.set_ylabel(variables[0] + ' (m)',fontsize=12)
    ax2.set_ylabel(variables[1] + ' (s)',fontsize=12)
    ax3.set_ylabel(variables[2] + ' (°)',fontsize=12)
    ax1.set_xlabel(' ')
    ax2.set_xlabel(' ')
    ax3.set_xlabel('Time')
    gs1.tight_layout(fig, h_pad=0.00001, rect=[0.05, 0.37, 0.62, []])


    #----------------
    # histogram
    gs2=gridspec.GridSpec(3,1)
    ax1=fig.add_subplot(gs2[0])
    ax2=fig.add_subplot(gs2[1])
    ax3=fig.add_subplot(gs2[2])

    hsim=np.reshape(bulk_sim[variables[0]].values,[-1,1])
    tsim=np.reshape(bulk_sim[variables[1]].values,[-1,1])
    dsim=np.reshape(bulk_sim[variables[2]].values,[-1,1])

    ax1.hist(hsim,np.linspace(0,6,30),density=True,label='Emulator',color='mediumpurple',alpha=0.9)
    ax2.hist(tsim,np.linspace(2,15,30),density=True,label='Emulator',color='mediumpurple',alpha=0.9)
    ax3.hist(dsim,np.linspace(0,360,30),density=True,label='Emulator',color='mediumpurple',alpha=0.9)

    ax1.hist(bulk_his[variables[0] ],np.linspace(0,6,30),density=True,label='Hindcast',color='lightcoral',alpha=0.5)
    ax2.hist(bulk_his[variables[1] ],np.linspace(2,15,30),density=True,label='Hindcast',color='lightcoral',alpha=0.5)
    ax3.hist(bulk_his[variables[2] ],np.linspace(0,360,30),density=True,label='Hindcast',color='lightcoral',alpha=0.5)
    plt.legend()
    ax1.set_xlabel(variables[0] + ' (m)')
    ax2.set_xlabel(variables[1] + ' (s)')
    ax3.set_xlabel(variables[2] + ' (°)')

    gs2.tight_layout(fig, h_pad=0.00001, rect=[0.6, 0.37, 0.8, []])

    #----------------
    # qq plot
    gs3=gridspec.GridSpec(3,1)
    ax1=fig.add_subplot(gs3[0])
    ax2=fig.add_subplot(gs3[1])
    ax3=fig.add_subplot(gs3[2])

    qt=np.linspace(0,1,51)
    ax1.plot([0, 6],[0, 6],':',color='palevioletred')
    ax1.plot(np.quantile(hsim,qt),np.quantile(bulk_his[variables[0]][bulk_his[variables[0]]>0],qt),'.-',color='navy')
    ax1.set_xlim([0,6]); ax1.set_ylim([0,6])
    ax1.set_xlabel('Emulator'); ax1.set_ylabel('Hindcast')
    ax1.grid(which='both',linestyle=':')

    ax2.plot([2, 16],[2, 16],':',color='palevioletred')
    ax2.plot(np.nanquantile(tsim,qt),np.nanquantile(bulk_his[variables[1]],qt),'.-',color='navy')
    ax2.set_xlim([2,16]); ax2.set_ylim([2,16])
    ax2.set_xlabel('Emulator'); ax2.set_ylabel('Hindcast')
    ax2.grid(which='both',linestyle=':')

    ax3.plot([0, 360],[0, 360],':',color='palevioletred')
    ax3.plot(np.nanquantile(dsim,qt),np.nanquantile(bulk_his[variables[2]],qt),'.-',color='navy')
    ax3.set_xlim([0,360]); ax3.set_ylim([0,360])
    ax3.set_xlabel('Emulator'); ax3.set_ylabel('Hindcast')
    ax3.grid(which='both',linestyle=':')
    #ax1.legend()
    gs3.tight_layout(fig, h_pad=0.00001, rect=[0.79, 0.37, 0.99, []])

    #----------------
    #%% RETURN PERIOD

    Emulator_y=bulk_sim.resample(time='1Y').max()
    hindcast_y=bulk_his.resample(time='1Y').max()

     # aux func for calculating rp time
    def t_rp(time_y):
        ny = len(time_y)
        return np.array([1/(1-(n/(ny+1))) for n in np.arange(1,ny+1)])

    # RP calculation, var sorting historical
    t_h = t_rp(hindcast_y.time.dt.year)

    # RP calculation, var sorting simulation
    t_s = t_rp(Emulator_y.time.dt.year)

    # hs
    gs1=gridspec.GridSpec(2,1)
    ax=fig.add_subplot(gs1[0])
    ax1=fig.add_subplot(gs1[1])

    ax.semilogx(t_h, np.sort(hindcast_y[variables[0]]), 'ok',color='lightcoral', markersize = 4, label = 'Historical', zorder=9,)
    #ax.semilogx(t_s, np.sort(np.nanmean(Emulator_y[variables[0]],axis=1)), '-',color='mediumpurple', linewidth = 2, label = 'Simulation (mean)',  zorder=8,)
    ax.semilogx(t_s, np.sort((Emulator_y[variables[0]])), '-',color='mediumpurple', linewidth = 2, label = 'Simulation ' + str(sim),  zorder=8,)
    #if opt=='maxmin':
    #    ax.fill_between(t_s,np.sort(np.nanmin(Emulator_y[variables[0]],axis=1)), np.sort(np.nanmax(Emulator_y[variables[0]],axis=1)), color='powderblue',alpha=0.7, label = 'Sim (Max-Min)')
    #elif opt=='ci90':
    #    ax.fill_between(t_s,np.sort(np.nanpercentile(Emulator_y[variables[0]],5,axis=1)), np.sort(np.nanpercentile(Emulator_y[variables[0]],95,axis=1)), color='powderblue',alpha=0.7, label = 'Confidence interval 95%') # mal
        #ax.fill_between(t_s,np.nanpercentile(np.sort(Emulator_y[variables[0]]),5,axis=1)), np.nanpercentile(np.sort(Emulator_y[variables[0]]),95,axis=1)), color='powderblue',alpha=0.7, label = 'Confidence interval 95%')
    #

    # customize axs
    # ax.set_title('Annual Maxima - Spectral Reconstrucion', fontweight='bold',fontsize=13)
    ax.set_ylabel(variables[0] + ' (m)',fontsize=12)
    ax.set_xlim(left=10**0, right=np.max(np.concatenate([t_h,t_s])))
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.grid(which='both',linestyle=':')
    ax.set_xticks([])
    ax.set_ylim([2.5,np.nanmax(hindcast_y[variables[0]])+0.5])

    # tp
    ax1.semilogx(t_h, np.sort(hindcast_y[variables[1]]), 'ok',color='lightcoral', markersize = 4, label = 'Historical', zorder=9,)
    #ax1.semilogx(t_s, np.sort(np.nanmean(Emulator_y[variables[1]],axis=1)), '-',color='mediumpurple', linewidth = 2, label = 'Simulation (mean)',  zorder=8,)
    ax1.semilogx(t_s, np.sort((Emulator_y[variables[1]])), '-',color='mediumpurple', linewidth = 2, label = 'Simulation ' + str(sim),  zorder=8,)
    #if opt=='maxmin':
    #    ax1.fill_between(t_s,np.sort(np.nanmin(Emulator_y.Tm,axis=1)), np.sort(np.nanmax(Emulator_y.Tm,axis=1)), color='powderblue',alpha=0.7, label = 'Sim (Max-Min)')
    #    ax1.set_ylim([np.nanmin(Emulator_y.Tm)-1,np.nanmax(Emulator_y.Tm)+1])
    #elif opt=='ci90':
    #    ax1.fill_between(t_s,np.sort(np.nanpercentile(Emulator_y.Tm,5,axis=1)), np.sort(np.nanpercentile(Emulator_y.Tm,95,axis=1)), color='powderblue',alpha=0.7, label = 'Confidence interval 95%')
    #    ax1.set_ylim([np.nanmin(Emulator_y.Tm)-1,np.nanmax(Emulator_y.Tm)+1])

    # customize axs
    ax.legend(loc='lower right',fontsize=12,ncol=3)
    ax1.set_xlabel('Return Period (years)',fontsize=12)
    ax1.set_ylabel(variables[1] + ' (s)',fontsize=12)
    ax1.set_xlim(left=10**0, right=np.max(np.concatenate([t_h,t_s])))

    ax1.tick_params(axis='both', which='both', top=True, right=True)
    ax1.grid(which='both',linestyle=':')

    gs1.tight_layout(fig, h_pad=0,rect=[0.05, 0, 1, 0.4])



def Plot_superpoint_spectrum_wt(sp_mean, sp_wt, figsize = [15, 15], title='DWT',vmin=-0.01, vmax=0.01,anomaly=False):
    '''
    Plots superpoint spectrum season averages

    sp - superpoint dataset
    '''

    # direction and frequency coordinates
    x = np.deg2rad(sp_mean.dir.values)
    y = sp_mean.freq.values

    # generate figure and gridspec for axes
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(7, 6)

    # plot each season
    for ix, wt in enumerate(range(len(sp_wt.wt))):

        # get each season energy
        if anomaly:
            z = np.sqrt(sp_wt.sel(wt = wt).efth.values) - np.sqrt(sp_mean.efth.values)
            cmap = 'seismic'
        else:
            z = np.sqrt(sp_wt.sel(wt = wt).efth.values)
            cmap = 'magma'

        # add axes and use spectrum axplot
        ax = fig.add_subplot(gs[ix], projection='polar')
        axplot_spectrum(ax, x, y, z, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(
            title + ': {0}'.format(wt+1),
            fontsize = 16,
            fontweight = 'bold',
            #pad = 20
        )


    return fig
