import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def axplot_basemap(lon1, lon2, lat1, lat2, alpha=1, resolution='50m', 
                   labels=True, cfill='silver', cocean='lightcyan'):
    '''
    Auxiliary cartopy plot to replicate Basemap functionality
    '''
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add features
    ax.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree())
    ax.coastlines(resolution=resolution)
    ax.add_feature(cfeature.LAND, facecolor=cfill, alpha=alpha)
    ax.add_feature(cfeature.OCEAN, facecolor=cocean)

    # Add gridlines
    gl = ax.gridlines(draw_labels=labels)
    gl.top_labels = False
    gl.right_labels = False

    return ax

def axplot_rectangle(ax, rect_lon1, rect_lon2, rect_lat1, rect_lat2,
                     color='dodgerblue', linewidth=5):
    'axes plot rectangle'
    ax.plot(
        [rect_lon1, rect_lon2, rect_lon2, rect_lon1, rect_lon1],
        [rect_lat1, rect_lat1, rect_lat2, rect_lat2, rect_lat1],
        '-',
        c=color,
        linewidth=linewidth,
        transform=ccrs.PlateCarree()
    )
    return ax

def Plot_target_area(rectangle=[], figsize=[20,7]):
    '''
    Plot the target area indicating the target islands.
    The input is the rectangle defining your target area.

    rectangle - [lon_ini, lon_end, lat_ini, lat_end] draws a rectangle
    '''

    # Generate figure
    fig = plt.figure(figsize=figsize)
    ax = axplot_basemap(rectangle[0], rectangle[1], rectangle[2], rectangle[3])

    plt.title('Target Area', fontweight='bold')

    # Plot rectangle if specified
    if len(rectangle) == 4:
        axplot_rectangle(ax, rectangle[0], rectangle[1], rectangle[2], rectangle[3])

    return fig

