
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import sys



def colors_snakes(n_colors=70):
     
    cm = plt.get_cmap('Paired')
    a=[cm(1.*i/n_colors) for i in range(n_colors)];
    cm = plt.get_cmap('Set3')
    b=[cm(1.*i/n_colors) for i in range(n_colors)];
    cm = plt.get_cmap('Accent')
    c=[cm(1.*i/n_colors) for i in range(n_colors)];
    cm = plt.get_cmap('Set2')
    d=[cm(1.*i/n_colors) for i in range(n_colors)];
    a=np.concatenate((a,b,c,d))
    np.random.shuffle(a) #Changes the order in a
    
    return a

def plot_partitions(spec, figsize=[18,9]):
    
    fig, axes= plt.subplots(3,figsize=figsize,sharex=True) 
    
    colors=['darkred','darkgreen','royalblue','crimson','plum','firebrick','gold','slateblue', 'yellow','mediumorchid','orange',
            'turquoise','pink','salmon','darkseagreen']

    for i in range(len(spec.part.values)):
        
        if spec.hs.isel(part=i).part.values==0:
            axes[0].plot(spec.time.values, spec.hs.isel(part=i).values,'.',color='grey',markersize=1, label='Sea')
            try:
                axes[2].plot(spec.time.values, spec.dir.isel(part=i).values,'.',color='grey',markersize=1)
            except:
                axes[2].plot(spec.time.values, spec.dpm.isel(part=i).values,'.',color='grey',markersize=1)
            axes[1].plot(spec.time.values, spec.tp.isel(part=i).values,'.',color='grey',markersize=1)
        else:
            if np.nanmin(spec.part)==0:
                i1=i-1
            else:
                i1=i
            axes[0].plot(spec.time.values, spec.hs.isel(part=i).values,'.',color=colors[i1],markersize=1, label='Swell: ' + str(spec.part.values[i]))
            axes[1].plot(spec.time.values, spec.tp.isel(part=i).values,'.',color=colors[i1],markersize=1)
            try:
                axes[2].plot(spec.time.values, spec.dir.isel(part=i).values,'.',color=colors[i1],markersize=1)
            except:
                axes[2].plot(spec.time.values, spec.dpm.isel(part=i).values,'.',color=colors[i1],markersize=1)

    axes[0].legend(ncol=len(spec.part), fontsize=15)
    axes[0].set_xlim([spec.time.values[0], spec.time.values[-1]])
    axes[0].set_ylabel('Hs (m)', fontsize=16)
    axes[1].set_ylabel('Tp (s)', fontsize=16)
    axes[2].set_ylabel('Dir (º)', fontsize=16)
    
    
def plot_snakes(spec, snakes, min_hs_plot=0, min_plot=0, figsize=[22,10], plot_ini=False):
    
    '''
    spec:         Spec over which to plot snakes, can be normalized or not
    snakes:       Matrix with dimensions part x time, with isolated swells positions
    min_hs_plot:  If defined, minimum hs to plot, if not all are plotted
    min_plot:     If defined, minimum number of swell times to plot, if not all are plotted
    
    '''

    mini = int(np.nanmin(snakes))
    maxi = int(np.nanmax(snakes))

    fig, axes= plt.subplots(3,figsize=figsize, sharex=True)
    for i in range(0,len(spec.part)):
        axes[0].plot(spec.time.values, spec.hs.isel(part=i).values,'k.',markersize=1,color='grey')
        axes[1].plot(spec.time.values, spec.tp.isel(part=i).values,'k.',markersize=1, color='grey')

        try:
            axes[2].plot(spec.time.values, spec.dir.isel(part=i).values,'k.',markersize=1, color='grey')
        except:
            axes[2].plot(spec.time.values, spec.dpm.isel(part=i).values,'k.',markersize=1, color='grey')

    a=colors_snakes(n_colors=70)
    axes[0].set_prop_cycle('color', a)
    axes[1].set_prop_cycle('color',a)
    axes[2].set_prop_cycle('color', a)

    for s in range(mini, maxi):
        s_p=np.where(snakes==s)
        
        if len(s_p[1])>min_plot: #Min number of points to consider a snake
            order=np.argsort(s_p[1])
            tt=s_p[1][order]
            value=s_p[0][order]

            if np.nanmax(spec.hs.values[value, tt])>min_hs_plot:
                axes[0].plot(np.sort(spec.time.values[tt]), spec.hs.values[value, tt], marker='.', markersize = 5)
                axes[1].plot(np.sort(spec.time.values[tt]), spec.tp.values[value, tt], marker='.', markersize = 5)
                axes[2].plot(np.sort(spec.time.values[tt]), spec.dpm.values[value, tt], '.', markersize = 5)

                if plot_ini==True:
                    axes[0].plot(np.sort(spec.time.values[tt])[0], spec.hs.values[value, tt][0], marker='*', markersize = 15)
                    axes[1].plot(np.sort(spec.time.values[tt])[0], spec.tp.values[value, tt][0], marker='*', markersize = 15)
                    axes[2].plot(np.sort(spec.time.values[tt])[0], spec.dpm.values[value, tt][0], marker='*', markersize = 15)
                    axes[0].grid()
                    axes[1].grid()
                    axes[2].grid()

    axes[0].set_xlim([spec.time.values[0], spec.time.values[tt[-1]]])
    axes[0].set_ylabel('Hs (m)', fontsize=16)
    axes[1].set_ylabel('Tp (s)', fontsize=16)
    axes[2].set_ylabel('Dir (º)', fontsize=16)


def plot_spectrum_hs(ax,x,y,z,z1=[], vmin=0, vmax=0.6,  vmin_z1=0, vmax_z1=0.3, ylim=0.49, size_point=5,
                     point_edge_color=None, alpha_bk=1, cmap='inferno', cmap_z1= 'inferno',
                     remove_axis=0, prob=None, prob_max=0.06, lw=5):

    if cmap=='RdBu_r':
        norm = mcolors.TwoSlopeNorm(0,vmin, vmax)
        p1=ax.pcolormesh(x,y,z,  cmap=plt.cm.RdBu_r, norm=norm, alpha=alpha_bk, linewidths=0.0000001) #vmin=vmin,vmax=vmax,
    else:
        p1=ax.pcolormesh(x,y,z, vmin=vmin, vmax=vmax,shading='flat', cmap=cmap, alpha=alpha_bk, linewidths=0.00000001)


    if len(z1):
        dx=(x[1]-x[0])/2
        dy=(y[1]-y[0])/2
        xx = x[1:]-dx,
        yy = y[1:]-dy
        xx,yy=np.meshgrid(xx,yy)

        p_z1=ax.scatter(xx,yy,size_point, z1, edgecolors=point_edge_color,vmin=vmin_z1,vmax=vmax_z1, cmap=cmap_z1)

    ax.set_theta_zero_location('N', offset=0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0,ylim)

    if prob:
        norm = Normalize(vmin=0, vmax=prob_max)
        cmap = cm.get_cmap('Blues')
        ax.spines['polar'].set_color(cmap(norm(prob)))
        ax.spines['polar'].set_linewidth(lw)

    if remove_axis:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(axis='y', colors='plum',labelsize=14,grid_linestyle=':',grid_alpha=0.75,grid_color='plum')
        ax.tick_params(axis='x', colors='purple',labelsize=14,pad=5,grid_linestyle=':',grid_alpha=0.75,grid_color='plum')
    ax.grid(color='plum', linestyle='--', linewidth=0.7,alpha=0.2)

    if len(z1):
        return p1, p_z1
    else:
        return p1


def plot_snakes_params(n_dwts,n_snakes,hs_snakes, annomaly=[], amp_dir=15,amp_t=5,t_max=30,prob_max=0.06, vmax_z1=1.2, tcs = False, figsize = [17, 13]):
    import matplotlib as mpl


    # bin dirs
    bin_dir = np.arange(0,360+amp_dir,amp_dir)

    # bin periods
    bin_t = np.arange(0.0,t_max+amp_t,amp_t)


    fig = plt.figure(figsize=figsize)
    if tcs:
        gs3=gridspec.GridSpec(7,6, hspace=0.05, wspace=0.05)
    else:
        gs3=gridspec.GridSpec(6,6, hspace=0.05, wspace=0.05)

    # mean snakes/dwt (for anomaly)
#    mean_prob = np.sum(n_snakes,axis=0)/np.sum(n_dwts)
    mean_prob=[]
    for wt in range(len(n_dwts)):
        n_wt = n_dwts[wt]
        n_snakes_wt = n_snakes[wt,:,:]
        mean_prob.append(n_snakes_wt/n_wt)

    mean_prob = np.nanmean(mean_prob, axis=0)


    for wt in range(len(n_dwts)):

        ax2=fig.add_subplot(gs3[wt],projection='polar')

        # select data for the dwt (all dirs and periods)
        n_wt = n_dwts[wt]
        n_snakes_wt = n_snakes[wt,:,:]
        hs_snakes_wt = hs_snakes[wt,:,:]


        # wt prob
        prob=n_wt/np.sum(n_dwts)

        # hs
        if annomaly:

            z1 = hs_snakes_wt - np.nanmean(hs_snakes, axis=0) #z1 plots points
            z = (n_snakes_wt/n_wt) - mean_prob # z is the background


            [p2,p_z1]=plot_spectrum_hs(ax2, np.deg2rad(bin_dir), bin_t, z, point_edge_color='Grey',  cmap='RdBu_r', vmin=-0.04, vmax=0.04,
                                alpha_bk=0.4, z1=z1, vmin_z1=-vmax_z1, vmax_z1=vmax_z1, cmap_z1='RdBu_r', size_point=18,
                                remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(bin_t))

        else:
            z1 = hs_snakes_wt #z1 plots points
            z = n_snakes_wt / n_wt # z is the background


            [p2,p_z1]=plot_spectrum_hs(ax2, np.deg2rad(bin_dir), bin_t, z, cmap='Greys', vmin=0, vmax=0.1, alpha_bk=0.4,
                                z1=z1, vmin_z1=0, vmax_z1=vmax_z1, cmap_z1='CMRmap_r', size_point=18,
                                remove_axis=1, prob=prob, prob_max=prob_max, ylim=np.nanmax(bin_t))



    gs3.tight_layout(fig, rect=[[], [], 0.78, []])

    gs4=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs4[0])
    plt.colorbar(p2,cax=ax0, extend='both')
    ax0.set_ylabel('nº Snakes/DWT', fontsize=14)
    gs4.tight_layout(fig, rect=[0.77, 0.1, 0.85, 0.9])

    gs5=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs5[0])
    plt.colorbar(p_z1,cax=ax0, extend='both')
    ax0.set_ylabel('Hs (m)', fontsize=14)
    gs5.tight_layout(fig, rect=[0.85, 0.1, 0.93, 0.9])

    gs6=gridspec.GridSpec(1,1)
    ax0=fig.add_subplot(gs6[0])
    norm = Normalize(vmin=0, vmax=0.02)
    cmap = cm.get_cmap('Blues')
    cb1 = mpl.colorbar.ColorbarBase(ax0, cmap=cmap, norm=norm, orientation='vertical', extend='both')
    cb1.set_label('Cluster Probability', fontsize=14)
    gs6.tight_layout(fig, rect=[0.93, 0.1, 1.02, 0.9])
    

def plot_his_sim_NumSnakes(n_swells, n_swells_sim):
    _faspect = 1.618
    _fsize = 9.8
    _fdpi = 128
    n_rows = 7
    n_cols = 6

    # plot figure
    fig = plt.figure(figsize=(_faspect*_fsize, _fsize))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.1, hspace=0.1)
    gr, gc = 0, 0
    for bmus in range(42):

        ax = plt.subplot(gs[gr, gc])

        # historical (from 0 to 41)
        n_swells_bmus = n_swells.where(n_swells.bmus==bmus, drop=True)
        n_swells_bmus_mean = n_swells_bmus.n_swells.mean(dim='time')

        # simulated (from 1 to 42)
        n_swells_sim_bmus = n_swells_sim.where(n_swells_sim.bmus==bmus+1, drop=True)
        n_swells_sim_bmus_mean = n_swells_sim_bmus.n_swells.mean(dim='time')


        ax.plot(n_swells_bmus_mean.n_dirs, n_swells_bmus_mean, '.-r', label='historical')
        ax.plot(n_swells_sim_bmus_mean.n_dirs, n_swells_sim_bmus_mean,'.-b', label='simulation')


        # wt text
        ax.text(0.87, 0.85, bmus+1, transform=ax.transAxes, fontweight='bold')


        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

        plt.yticks([]);
        plt.suptitle('mean number of snakes/day for each DWT and direction')
        plt.ylim(0,.9)


        if gc == 1 and gr == 0:
            plt.legend(
                bbox_to_anchor=(1, 1),
                bbox_transform=fig.transFigure,
            )

        if gc == 1:
            plt.ylabel('nº swells/day')
            plt.yticks([0.3,.6,.9]);

        if gr >= 5:
            plt.xlabel('dirs')



def plot_sim_snakes(time_h, hs_swell, tp_swell, dir_swell, sea_params_sim):

    NUM_COLORS = 100
    cm = plt.get_cmap('Paired')
    a=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)];
    cm = plt.get_cmap('Set3')
    b=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)];
    cm = plt.get_cmap('Accent')
    c=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)];
    cm = plt.get_cmap('Set2')
    d=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)];
    colors=np.concatenate((a,b,c,d))

    fig = plt.figure(figsize=[18.5,9])
    gs1=gridspec.GridSpec(3,1)
    ax1=fig.add_subplot(gs1[0])
    ax2=fig.add_subplot(gs1[1],sharex=ax1)
    ax3=fig.add_subplot(gs1[2],sharex=ax1)

    for r in range(len(hs_swell)):

        qq = np.random.randint(0, np.shape(colors)[0],1)

        ax1.plot(time_h, hs_swell[r,:],'.-',markersize=1, color = (colors[qq][0][0], colors[qq][0][1], colors[qq][0][2]))
        ax2.plot(time_h, tp_swell[r,:],'.-',markersize=1, color = (colors[qq][0][0], colors[qq][0][1], colors[qq][0][2]))
        ax3.plot(time_h, dir_swell[r,:],'.-',markersize=1,color = (colors[qq][0][0], colors[qq][0][1], colors[qq][0][2]))

        ind_hs_max = np.nanargmax(hs_swell[r,:])
        ind_tp = np.where(~np.isnan(hs_swell[r,:]))[0][0]
        ax1.plot(time_h[ind_hs_max],hs_swell[r,ind_hs_max],'.',markersize=8,color=(colors[qq][0][0], colors[qq][0][1], colors[qq][0][2]))
        ax2.plot(time_h[ind_tp],    tp_swell[r,ind_tp],'.',markersize=8,color=(colors[qq][0][0], colors[qq][0][1], colors[qq][0][2]))
        ax3.plot(time_h[ind_hs_max],dir_swell[r,ind_hs_max],'.',markersize=6,color=(colors[qq][0][0], colors[qq][0][1], colors[qq][0][2]))

        ax1.set_ylim(0,4); ax2.set_ylim(2,29)
        ax1.set_ylabel('Hs (m)',fontsize=13)
        ax2.set_ylabel('Tp (s)',fontsize=13)
        ax3.set_ylabel('Dir (°)',fontsize=13)

    ax1.set_xlim(time_h[0], time_h[-1])

    #We also plot the sea on top
    ax1.plot(sea_params_sim.time, sea_params_sim.Hs_sea,':',color='black',linewidth=1.3,alpha=0.6)
    ax2.plot(sea_params_sim.time, sea_params_sim.Tp_sea,':',color='black',linewidth=1.5,alpha=0.6)
    ax3.plot(sea_params_sim.time, sea_params_sim.Dir_sea,'.',color='black',markersize=0.3,alpha=0.75)
    plt.suptitle('simulated snakes' ,fontsize=14,fontweight='bold')
