
import numpy as np
import xarray as xr


def spatial_gradient(xdset, var_name):
    '''
    Calculate spatial gradient

    xdset:
        (longitude, latitude, time), var_name

    returns xdset with new variable "var_name_gradient"
    '''

    # TODO:check/ ADD ONE ROW/COL EACH SIDE
    var_grad = np.zeros(xdset[var_name].shape)

    Mx = len(xdset.longitude)
    My = len(xdset.latitude)
    lat = xdset.latitude.values

    for it in range(len(xdset.time)):
        var_val = xdset[var_name].isel(time=it).values

        # calculate gradient (matrix)
        m_c = var_val[1:-1,1:-1]
        m_l = np.roll(var_val, -1, axis=1)[1:-1,1:-1]
        m_r = np.roll(var_val, +1, axis=1)[1:-1,1:-1]
        m_u = np.roll(var_val, -1, axis=0)[1:-1,1:-1]
        m_d = np.roll(var_val, +1, axis=0)[1:-1,1:-1]
        m_phi = np.pi*np.abs(lat)/180.0
        m_phi = m_phi[1:-1]

        dpx1 = (m_c - m_l)/np.cos(m_phi[:,None])
        dpx2 = (m_r - m_c)/np.cos(m_phi[:,None])
        dpy1 = m_c - m_d
        dpy2 = m_u - m_c

        vg = (dpx1**2+dpx2**2)/2 + (dpy1**2+dpy2**2)/2
        var_grad[it, 1:-1, 1:-1] = vg

        # calculate gradient (for). old code
        #for i in range(1, Mx-1):
        #    for j in range(1, My-1):
        #        phi = np.pi*np.abs(lat[j])/180.0
        #        dpx1 = (var_val[j,i]   - var_val[j,i-1]) / np.cos(phi)
        #        dpx2 = (var_val[j,i+1] - var_val[j,i])   / np.cos(phi)
        #        dpy1 = (var_val[j,i]   - var_val[j-1,i])
        #        dpy2 = (var_val[j+1,i] - var_val[j,i])
        #        var_grad[it, j, i] = (dpx1**2+dpx2**2)/2 + (dpy1**2+dpy2**2)/2

    # store gradient
    xdset['{0}_gradient'.format(var_name)]= (
        ('time', 'latitude', 'longitude'), var_grad)

    return xdset
