#-------------------------------------------------------------------------------
# Name:        spatial_interpolation.py
# Purpose:     Useful classes and functions for spatial interpolation.
#
# Author:      James Sample
#
# Created:     02/11/2016
# Copyright:   (c) James Sample and NIVA, 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
""" The ECCO-DomQua project requires spatial interpolation of various datasets.
    Code for this has been developed and tested in 
    spatial_interpolation_2.ipynb. This file gathers the main classes and 
    functions together into a single file, which can be imported into e.g.
    iPython notebooks for easy reuse.
"""
from __future__ import division
import numpy as np
from scipy.spatial import cKDTree as KDTree

def read_geotiff(geotiff_path):
    """ Reads a GeoTiff file to a numpy array.
    
    Args:
        geotiff_path Path to file
    
    Returns:
        Tuple: (array, NDV, (xmin, xmax, ymin, ymax))
        No data values in the array are set to np.nan
    """
    from osgeo import gdal, gdalconst
    import numpy as np, numpy.ma as ma, sys

    # Register drivers
    gdal.AllRegister()

    # Process the file with GDAL
    ds = gdal.Open(geotiff_path, gdalconst.GA_ReadOnly)
    if ds is None:
        print 'Could not open ' + geotiff_path
        sys.exit(1)

    # Dataset properties
    geotransform = ds.GetGeoTransform()
    originX = geotransform[0]   # Origin is top-left corner
    originY = geotransform[3]   # i.e. (xmin, ymax)
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    
    # Calculate extent
    xmin = int(originX)
    xmax = int(originX + cols*pixelWidth)
    ymin = int(originY + rows*pixelHeight)
    ymax = int(originY)

    # Read band 1
    band = ds.GetRasterBand(1)
    
    # Get NDV
    no_data_val = band.GetNoDataValue()   

    # Read the data to an array
    data = band.ReadAsArray()
    
    # Set NDV to np.nan
    data[data==no_data_val] = np.nan

    # Flip. The data was flipped when it was created (see array_to_gtiff, 
    # above), so need to flip it back. Not sure why this is necessary? 
    # Beware in future!
    data = data[::-1,:]
    
    # Close the dataset
    ds = None

    return (data, no_data_val, (xmin, xmax, ymin, ymax))
    
def array_to_gtiff(xmin, ymax, cell_size, out_path, data_array,
                   proj4_str, no_data_value=-9999):
    """ Save numpy array as GeoTiff (in a projected co-ordinate system).
    
    Args:
        xmin:          Minimum x value in metres
        ymax:          Maximum y value in metres
        cell_size:     Grid cell size in metres
        out_path:      Path to GeoTiff
        data:          Array to save 
        proj4_str      proj.4 string defining the projection
        no_data_value: Value to use to represent no data 
        
    Returns:
        None. Array is saved to specified path.
    """
    # Import modules
    import gdal, gdalconst, osr

    # Explicitly set NDV
    data_array[np.isnan(data_array)] = no_data_value
    
    # Flip. Not sure why this is necessary? Without it the 
    # output grid is upside down! Haven't had this problem before,
    # but this seems to work here. Beware in future!
    data_array = data_array[::-1,:]

    # Get array shape
    cols = data_array.shape[1]
    rows = data_array.shape[0]

    # Get driver
    driver = gdal.GetDriverByName('GTiff')

    # Create a new raster data source
    out_ds = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)

    # Get spatial reference
    sr = osr.SpatialReference()
    sr.ImportFromProj4(proj4_str)
    sr_wkt = sr.ExportToWkt()

    # Write metadata
    # (xmin, cellsize, 0, ymax, 0, -cellsize)
    out_ds.SetGeoTransform((int(xmin), cell_size, 0.0, 
                            int(ymax), 0.0, -cell_size))  
    out_ds.SetProjection(sr_wkt)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(-9999)
    out_band.WriteArray(data_array)

    # Tidy up
    del out_ds, out_band
    
def idw_interp(pts, z, xi, yi, n_near=8, p=1):
    """ Simplified interface to Invdisttree class.
    
    Args:
        pts    2D array of (x, y) pairs for known points
        z      1D array of values to interpolate
        xi     1D array of x values to interpolate 
        yi     1D array of y values to interpolate  
        n_near The number of nearest neighbours to consider
        p      Power defining rate at which weights decrease with distance
    """  
    # Build interpolator
    invdisttree = Invdisttree(pts, z)
    
    # Build list of co-ords to interpolate
    xx, yy = np.meshgrid(xi, yi)    
    pts_i = np.array(zip(xx.flatten(), yy.flatten()))
    
    # Perform interpolation
    interpol = invdisttree(pts_i, nnear=n_near, p=p)
    
    # Reshape output
    zi = interpol.reshape((len(yi), len(xi)))
    
    return zi
    
class Invdisttree:
    """ The code for this class is taken from:
        
    http://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python
    
    inverse-distance-weighted interpolation using KDTree.
    
    invdisttree = Invdisttree( X, z )  -- data points, values
    interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

    How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

    p=1, p=2 ?
        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3)
            / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.

    Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std().

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.
    """
    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]