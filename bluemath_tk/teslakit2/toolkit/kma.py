#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip
import numpy as np
import xarray as xr
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans


def sort_cluster_gen_corr_end(centers, dimdim):
    '''
    SOMs alternative
    '''

    # get dimx, dimy
    dimy = np.floor(np.sqrt(dimdim)).astype(int)
    dimx = np.ceil(np.sqrt(dimdim)).astype(int)

    if not np.equal(dimx*dimy, dimdim):
        # TODO: RAISE ERROR
        pass

    dd = distance_matrix(centers, centers)
    qx = 0
    sc = np.random.permutation(dimdim).reshape(dimy, dimx)

    # get qx
    for i in range(dimy):
        for j in range(dimx):

            # row F-1
            if not i==0:
                qx += dd[sc[i-1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i-1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i-1,j+1], sc[i,j]]

            # row F
            if not j==0:
                qx += dd[sc[i,j-1], sc[i,j]]

            if not j+1==dimx:
                qx += dd[sc[i,j+1], sc[i,j]]

            # row F+1
            if not i+1==dimy:
                qx += dd[sc[i+1,j], sc[i,j]]

                if not j==0:
                    qx += dd[sc[i+1,j-1], sc[i,j]]

                if not j+1==dimx:
                    qx += dd[sc[i+1,j+1], sc[i,j]]

    # test permutations
    q=np.inf
    go_out = False
    for i in range(dimdim):
        if go_out:
            break

        go_out = True

        for j in range(dimdim):
            for k in range(dimdim):
                if len(np.unique([i,j,k]))==3:

                    u = sc.flatten('F')
                    u[i] = sc.flatten('F')[j]
                    u[j] = sc.flatten('F')[k]
                    u[k] = sc.flatten('F')[i]
                    u = u.reshape(dimy, dimx, order='F')

                    f=0
                    for ix in range(dimy):
                        for jx in range(dimx):

                            # row F-1
                            if not ix==0:
                                f += dd[u[ix-1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix-1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix-1,jx+1], u[ix,jx]]

                            # row F
                            if not jx==0:
                                f += dd[u[ix,jx-1], u[ix,jx]]

                            if not jx+1==dimx:
                                f += dd[u[ix,jx+1], u[ix,jx]]

                            # row F+1
                            if not ix+1==dimy:
                                f += dd[u[ix+1,jx], u[ix,jx]]

                                if not jx==0:
                                    f += dd[u[ix+1,jx-1], u[ix,jx]]

                                if not jx+1==dimx:
                                    f += dd[u[ix+1,jx+1], u[ix,jx]]

                    if f<=q:
                        q = f
                        sc = u

                        if q<=qx:
                            qx=q
                            go_out=False

    return sc.flatten('F')

def kma_simple(xds_PCA, num_clusters, repres=0.95):
    '''
    KMeans Classification for PCA data

    xds_PCA       - Principal Component Analysis  (xarray.Dataset)
                    (n_components, n_components) PCs
                    (n_components, n_features) EOFs
                    (n_components, ) variance
    num_clusters  - number of clusters desired from classification
    repres        -

    returns a xarray.Dataset containing KMA data
    '''

    # PCA data
    variance = xds_PCA.variance.values[:]
    EOFs = xds_PCA.EOFs.values[:]
    PCs = xds_PCA.PCs.values[:]

    var_anom_std = xds_PCA.var_anom_std.values[:]
    var_anom_mean = xds_PCA.var_anom_mean.values[:]
    time = xds_PCA.time.values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    print('Number of PCs: ' + str(len(variance)))
    print('Number of PCs explaining ' + str(repres*100) + '% of the EV is: ' + str(nterm))

    PCsub = PCs[:, :nterm+1]
    EOFsub = EOFs[:nterm+1, :]

    # KMEANS
    kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(PCsub)

    # groupsize
    _, group_size = np.unique(kma.labels_, return_counts=True)

    # groups
    d_groups = {}
    for k in range(num_clusters):
        d_groups['{0}'.format(k)] = np.where(kma.labels_==k)
    # TODO: STORE GROUPS WITHIN OUTPUT DATASET

    # centroids
    centroids = np.dot(kma.cluster_centers_, EOFsub)

    # km, x and var_centers
    km = np.multiply(
        centroids,
        np.tile(var_anom_std, (num_clusters, 1))
    ) + np.tile(var_anom_mean, (num_clusters, 1))

    # sort kmeans
    kma_order = np.argsort(np.mean(-km, axis=1))

    # reorder clusters: bmus, km, cenEOFs, centroids, group_size
    sorted_bmus = np.zeros((len(kma.labels_),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(kma.labels_ == kma_order[i])
        sorted_bmus[posc] = i
    sorted_km = km[kma_order]
    sorted_cenEOFs = kma.cluster_centers_[kma_order]
    sorted_centroids = centroids[kma_order]
    sorted_group_size = group_size[kma_order]

    return xr.Dataset(
        {
            'bmus': (('n_pcacomp'), sorted_bmus.astype(int)),
            'cenEOFs': (('n_clusters', 'n_features'), sorted_cenEOFs),
            'centroids': (('n_clusters','n_pcafeat'), sorted_centroids),
            'Km': (('n_clusters','n_pcafeat'), sorted_km),
            'group_size': (('n_clusters'), sorted_group_size),

            # PCA data
            'PCs': (('n_pcacomp','n_features'), PCsub),
            'variance': (('n_pcacomp',), variance),
            'time': (('n_pcacomp',), time),
        }
    )

def slp_kmeans_clustering(PCA, repres=0.95, num_clusters=[], min_data=20):
 # TODO: funcion kma simple de Laura, input de numero minimo de datos por cluster,
 #  unificar con kma_simple de Nico
    import sys

    variance = PCA['variance'].values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    print('Number of PCs: ' + str(len(variance)))
    print('Number of PCs explaining ' + str(repres*100) + '% of the EV is: ' + str(nterm))

    nterm = nterm+1

    num, it=0,0

    while np.nanmin(num)<min_data:

        sys.stdout.write('\rIteration: %d' %it)
        sys.stdout.flush()

        kma = KMeans(n_clusters=num_clusters,init='k-means++',n_init=10,tol=0.0001) #80
        bmus = kma.fit_predict(PCA.PCs[:,:nterm])
        centers= kma.cluster_centers_
        num=[]
        for i in range(num_clusters):
            num=np.append(num,len(np.where(bmus==i)[0]))
        it=it+1

    #pk.dump(kma, open(os.path.join(path_main, 'Results', site , 'SLP_KMA_' + site + '.pkl'),"wb"))

    print('Number of sims: ' + str(kma.n_iter_))
    _, group_size = np.unique(bmus, return_counts=True)
    print('Minimum number of data: ' + str(np.nanmin(group_size)))

    # km, x and var_centers
    kma_order = np.argsort(np.mean(-centers, axis=1))

    sorted_bmus = np.zeros((len(bmus),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(bmus == kma_order[i])
        sorted_bmus[posc] = i

    bmus=sorted_bmus


    return xr.Dataset(
        {
            'sorted_bmus':(('n_components'), bmus),
            'kma_order':(('n_clusters'),kma_order),
            'cluster':(('n_clusters'),np.arange(num_clusters)),
            'cenEOFs':(('n_clusters','n_features'), centers),

        }
    )



def kma_regression_guided(
    xds_PCA, xds_Yregres, num_clusters,
    repres=0.95, alpha=0.5, min_group_size=None):
    '''
    KMeans Classification for PCA data: regression guided

    xds_PCA         - Principal Component Analysis  (xarray.Dataset)
                        (n_components, n_components) PCs
                        (n_components, n_features) EOFs
                        (n_components, ) variance
    xds_Yregres     - Simple multivariate regression  (xarray.Dataset)
                        (time, vars) Ym
    num_clusters    - number of clusters desired from classification 
    repres          -
    alpha           -
    min_group_size  - minimun number of samples to accept classification  
    '''

    # PCA data
    variance = xds_PCA['variance'].values[:]
    EOFs = xds_PCA['EOFs'].values[:]
    PCs = xds_PCA['PCs'].values[:]

    # Yregres data
    Y = xds_Yregres['Ym'].values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    print('Number of PCs: ' + str(len(variance)))
    print('Number of PCs explaining ' + str(repres*100) + '% of the EV is: ' + str(nterm))

    nterm = nterm+1
    PCsub = PCs[:, :nterm]

    # append Yregres data to PCs
    data = np.concatenate((PCsub, Y), axis=1)
    data_std = np.std(data, axis=0)
    data_mean = np.mean(data, axis=0)

    # normalize but keep PCs weigth
    data_norm = np.ones(data.shape)*np.nan
    for i in range(PCsub.shape[1]):
        data_norm[:,i] = np.divide(data[:,i]-data_mean[i], data_std[0])
    for i in range(PCsub.shape[1],data.shape[1]):
        data_norm[:,i] = np.divide(data[:,i]-data_mean[i], data_std[i])

    # apply alpha (PCs - Yregress weight)
    data_a = np.concatenate(
        ((1-alpha)*data_norm[:,:nterm],
         alpha*data_norm[:,nterm:]),
        axis=1
    )

    # KMeans
    keep_iter = True
    count_iter = 0
    while keep_iter:
        # n_init: number of times KMeans runs with different centroids seeds
        #kma = KMeans(
        #    n_clusters = num_clusters,
        #    init='random', n_init=30, max_iter=500,
        #    n_jobs=-1
        #).fit(data_a)

        # much faster KMeans algorithm
        kma = MiniBatchKMeans(
            n_clusters=num_clusters,
            n_init=10,
            max_iter=500
        ).fit(data_a)

        # check minimun group_size
        group_keys, group_size = np.unique(kma.labels_, return_counts=True)

        # sort output
        group_k_s = np.column_stack([group_keys, group_size])
        group_k_s = group_k_s[group_k_s[:,0].argsort()]  # sort by cluster num

        if not min_group_size:
            keep_iter = False

        else:
            # keep iterating?
            keep_iter1 = np.where(group_k_s[:,1] < min_group_size)[0].any()
            keep_iter2 = len(group_keys)!= num_clusters
            keep_iter = keep_iter1 or keep_iter2
            count_iter += 1

            # log kma iteration
            for rr in group_k_s:
                if rr[1] < min_group_size:
                    print('  c: {0} - s: {1}'.format(rr[0], rr[1]))
            print('total attemps: ', count_iter)
            print()

    # groups
    d_groups = {}
    for k in range(num_clusters):
        d_groups['{0}'.format(k)] = np.where(kma.labels_==k)
    # TODO: STORE GROUPS WITHIN OUTPUT DATASET    

    # centroids
    centroids = np.zeros((num_clusters, data.shape[1]))
    for k in range(num_clusters):
        centroids[k,:] = np.mean(data[d_groups['{0}'.format(k)],:], axis=1)

    # sort kmeans
    kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)

    bmus_corrected = np.zeros((len(kma.labels_),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(kma.labels_==kma_order[i])
        bmus_corrected[posc] = i

    # reorder centroids
    sorted_cenEOFs = kma.cluster_centers_[kma_order,:]
    sorted_centroids = centroids[kma_order,:]

    return xr.Dataset(
        {
            # KMA data
            'bmus': (('n_components',), kma.labels_),
            'cenEOFs': (('n_clusters', 'n_features'), kma.cluster_centers_),
            'centroids': (('n_clusters','n_features'), centroids),
            'group_size': (('n_clusters'), group_k_s[:,1]),

            # sorted KMA data
            'sorted_order': (('n_clusters'), kma_order),
            'sorted_bmus': (('n_components'), bmus_corrected.astype(int)),
            'sorted_cenEOFs': (('n_clusters', 'n_features'), sorted_cenEOFs),
            'sorted_centroids': (('n_clusters','n_features'), sorted_centroids),

        },
        attrs = {
            'method': 'regression guided',
            'alpha': alpha,
        }
    )

