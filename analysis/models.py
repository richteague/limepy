"""
Read in different models and grid the data for use with analysemodel.py.

    diskmodel - assumes an azimuthally symmetric disk around the z axis.
    slabmodel - assumes a slab with all gradients in the z direction.

"""

import os
import warnings
import numpy as np
from astropy.io import fits
import scipy.constants as sc
from scipy.interpolate import griddata


class diskmodel:

    def __init__(self, path, verbose=False, mindens=1e3, **kwargs):
        """
        Read in the disk model and prune the points.
        """
        self.path = path
        self.aux = os.path.dirname(os.path.realpath(__file__))
        self.aux = self.aux.replace('analysis', 'aux/')
        self.mindens = mindens
        self.depletion = kwargs.get('depletion', 1.0)
        self.verbose = verbose
        if not self.verbose:
            warnings.simplefilter('ignore')

        # Currently only important are the grid [1] and level populations [4]
        # from the grid. Both [2] and [3] are for the Delanunay triangulation
        # and can thus be ignored.

        self.hdu = fits.open(self.path)
        self.grid = self.hdu[1]
        if self.verbose:
            for c in self.grid.columns:
                print c
            print('\n')
        self.names = self.grid.columns.names

        # Coordinates. Remove all the sink particles and convert to au. The
        # native system is cartesian, (x, y, z). Also convert them into
        # spherical polar coordinates, (r, p, t).

        self.notsink = ~self.grid.data['IS_SINK']
        self.xvals = self.grid.data['x1'][self.notsink] / sc.au
        self.yvals = self.grid.data['x2'][self.notsink] / sc.au
        self.zvals = self.grid.data['x3'][self.notsink] / sc.au
        self.rvals = np.hypot(self.yvals, self.xvals)
        self.pvals = np.arctan2(self.yvals, self.xvals)
        self.tvals = np.arctan2(self.zvals, self.rvals)

        # Physical properties at each cell. If dtemp == -1, then use gtemp,
        # this allows us to calculate the dust continuum for the line emission.

        self.gtemp = self.grid.data['TEMPKNTC'][self.notsink]
        self.dtemp = self.grid.data['TEMPDUST'][self.notsink]
        self.dtemp = np.where(self.dtemp == -1, self.gtemp, self.dtemp)

        # Assume that the densities are only ever H2 or [oH2, pH2]. If the
        # latter, allow density to be the sum. Individual values can still be
        # accessed through _density.

        self.ndens = len([n for n in self.names if 'DENSITY' in n])
        if self.ndens > 1 and self.verbose:
            print('Assuming DENSITY1 and DENSITY2 are oH2 and pH2.')
        self._dens = {d: self.grid.data['DENSITY%d' % (d+1)][self.notsink]
                      for d in range(self.ndens)}
        self.dens = np.sum([self._dens[k] for k in range(self.ndens)], axis=0)

        # Include the other physical properties.

        self.nabun = len([n for n in self.names if 'ABUNMOL' in n])
        if self.nabun > 1:
            raise NotImplementedError()
        self.abun = self.grid.data['ABUNMOL1'][self.notsink]
        self.velo = np.array([self.grid.data['VEL%d' % i][self.notsink]
                              for i in [1, 2, 3]])
        self.turb = self.grid.data['TURBDPLR'][self.notsink]

        # Mask out all points with a total density of <= min_density, with a
        # default of 10^4. Include the depletion of the emitting molecule.

        self.dmask = self.dens > kwargs.get('min_density', 1e3)
        self.xvals = self.xvals[self.dmask]
        self.yvals = self.yvals[self.dmask]
        self.zvals = self.zvals[self.dmask]
        self.rvals = self.rvals[self.dmask]
        self.pvals = self.pvals[self.dmask]
        self.tvals = self.tvals[self.dmask]
        self.gtemp = self.gtemp[self.dmask]
        self.dtemp = self.dtemp[self.dmask]
        self.dens = self.dens[self.dmask]
        self.abun = self.abun[self.dmask] * self.depletion
        self.turb = self.turb[self.dmask]

        # Excitation properties. Remove all the sink particles.

        pops = self.hdu[4].data.T
        idxs = [i for i, b in enumerate(self.notsink) if not b]
        self.levels = np.delete(pops, idxs, axis=1)
        idxs = [i for i, b in enumerate(self.dmask) if not b]
        self.levels = np.delete(self.levels, idxs, axis=1)

        # -- Gridding Options --
        #
        # There are three options here. One can provide the axes either
        # as an array, otherwise the grids are generated depending on the
        # points of the model.
        # By default the grid is logarithmic in the vertical direction but
        # linear in the radial direction with 500 points in each, however
        # these are customisable.

        grids = kwargs.get('grids', None)
        if grids is None:
            self.xgrid, self.ygrid = self.estimate_grids(**kwargs)
        else:
            try:
                self.xgrid, self.ygrid = grids
            except ValueError:
                self.xgrid = grids
                self.ygrid = grids
            except:
                raise ValueError('grids = [xgrid, ygrid].')
        self.xpnts = self.xgrid.size
        self.ypnts = self.ygrid.size

        # With the grids, grid the parameters and store them in a dictionary.
        # Only read in the first (by default) 5 energy levels, but this can be
        # increased later with a call to self.grid_levels(j_max).

        self.method = kwargs.get('method', 'linear')
        if self.verbose:
            print('Beginning gridding using %s interpolation.' % self.method)
            if self.method == 'nearest':
                print('Warning: neartest may produce unwanted features.')

        self.gridded = {}
        self.gridded['dens'] = self.grid_param(self.dens, self.method)
        self.gridded['gtemp'] = self.grid_param(self.gtemp, self.method)
        self.gridded['dtemp'] = self.grid_param(self.dtemp, self.method)
        self.gridded['abun'] = self.grid_param(self.abun, self.method)
        self.gridded['turb'] = self.grid_param(self.turb, self.method)
        self.gridded['levels'] = {}
        self.grid_levels(kwargs.get('nlevels', 5))

        return

    def grid_levels(self, nlevels):
        """Grid the specified energy levels."""
        for j in np.arange(nlevels):
            if j in self.gridded['levels'].keys():
                continue
            self.gridded['levels'][j] = self.grid_param(self.levels[j],
                                                        self.method)
        self.jmax = max(self.gridded['levels'].keys())
        if self.verbose:
            print('Gridded the first %d energy levels.' % (self.jmax))
            print('Use self.grid_levels() to read in more.\n')
        return

    def grid_param(self, param, method='linear'):
        """Return a gridded version of param."""
        return griddata((self.rvals, self.zvals),
                        param, (self.xgrid[None, :], self.ygrid[:, None]),
                        method=method, fill_value=0.0)

    def estimate_grids(self, **kwargs):
        """Return grids based on points in the model."""
        npts = kwargs.get('npts', 500)
        xpts = kwargs.get('xnpts', npts)
        ypts = kwargs.get('ynpts', npts)
        xmin = self.rvals.min()
        xmax = self.rvals.max()
        if kwargs.get('logx', False):
            xgrid = np.logspace(np.log10(xmin), np.log10(xmax), xpts)
            if self.verbose:
                print('Made the xgrid logarithmic.')
        else:
            xgrid = np.linspace(xmin, xmax, xpts)
            if self.verbose:
                print('Made the xgrid linear.')
        ymin = abs(self.zvals).min()
        ymax = abs(self.zvals).max() * 1.05
        if kwargs.get('logy', True):
            ygrid = np.logspace(np.log10(ymin), np.log10(ymax), ypts / 2)
            ygrid = np.hstack([-ygrid[::-1], ygrid])
            if self.verbose:
                print('Made the ygrid logarithmic.')
        else:
            ygrid = np.linspace(-ymax, ymax, ypts)
            if self.verbose:
                print ('Made the ygrid linear.')
        return xgrid, ygrid


class slabmodel:

    def __init__(self, path, verbose=False, mindens=1e3, **kwargs):
        """
        Read in the slab model and prune the points.
        """
        self.path = path
        self.aux = os.path.dirname(os.path.realpath(__file__))
        self.aux = self.aux.replace('analysis', 'aux/')
        self.mindens = mindens
        self.verbose = verbose
        if not self.verbose:
            warnings.simplefilter('ignore')

        # Read in the .fits data. HDU[1] is the grid and HDU[4] are the level
        # populations. [2] and [3] can be ignored.

        self.hdu = fits.open(self.path)
        self.grid = self.hdu[1]
        if self.verbose:
            for c in self.grid.columns:
                print c
            print('\n')
        self.names = self.grid.columns.names

        # Remove all the sink particles and convert units to [au].

        self.notsink = ~self.grid.data['IS_SINK']
        self.xvals = self.grid.data['x1'][self.notsink] / sc.au
        self.yvals = self.grid.data['x2'][self.notsink] / sc.au
        self.zvals = self.grid.data['x3'][self.notsink] / sc.au

        # Extract the physical properties. Assume that the densities are only
        # ever H2 or [oH2, pH2]. If the latter, allow density to be the sum.
        # Individual values can still be accessed through _density.

        self.gtemp = self.grid.data['TEMPKNTC'][self.notsink]
        self.dtemp = self.grid.data['TEMPDUST'][self.notsink]
        self.dtemp = np.where(self.dtemp == -1, self.gtemp, self.dtemp)

        self.ndens = len([n for n in self.names if 'DENSITY' in n])
        if self.ndens > 1 and self.verbose:
            print('Assuming DENSITY1 and DENSITY2 are oH2 and pH2.')
        self._dens = {d: self.grid.data['DENSITY%d' % (d+1)][self.notsink]
                      for d in range(self.ndens)}
        self.dens = np.sum([self._dens[k] for k in range(self.ndens)], axis=0)

        self.nabun = len([n for n in self.names if 'ABUNMOL' in n])
        if self.nabun > 1:
            raise NotImplementedError()
        self.abun = self.grid.data['ABUNMOL1'][self.notsink]
        self.velo = np.array([self.grid.data['VEL%d' % i][self.notsink]
                              for i in [1, 2, 3]])
        self.turb = self.grid.data['TURBDPLR'][self.notsink]

        # Remove all particles that fall below the minimum density.

        self.dmask = self.dens > self.mindens
        self.xvals = self.xvals[self.dmask]
        self.yvals = self.yvals[self.dmask]
        self.zvals = self.zvals[self.dmask]
        self.gtemp = self.gtemp[self.dmask]
        self.dtemp = self.dtemp[self.dmask]
        self.dens = self.dens[self.dmask]
        self.abun = self.abun[self.dmask]
        self.turb = self.turb[self.dmask]

        # Remove all the particles that are |x_i| > rmax.

        self.rmax = kwargs.get('rmax', 20)
        self.rmask = np.where(abs(self.xvals) > self.rmax, 1, 0)
        self.rmask += np.where(abs(self.yvals) > self.rmax, 1, 0)
        self.rmask += np.where(abs(self.zvals) > self.rmax, 1, 0)
        self.rmask = np.where(self.rmask == 0, True, False)
        self.xvals = self.xvals[self.rmask]
        self.yvals = self.yvals[self.rmask]
        self.zvals = self.zvals[self.rmask]
        self.gtemp = self.gtemp[self.rmask]
        self.dtemp = self.dtemp[self.rmask]
        self.dens = self.dens[self.rmask]
        self.abun = self.abun[self.rmask]
        self.turb = self.turb[self.rmask]

        # Excitation properties. Remove all the sink particles.

        pops = self.hdu[4].data.T
        idxs = [i for i, b in enumerate(self.notsink) if not b]
        self.levels = np.delete(pops, idxs, axis=1)
        idxs = [i for i, b in enumerate(self.dmask) if not b]
        self.levels = np.delete(self.levels, idxs, axis=1)
        idxs = [i for i, b in enumerate(self.rmask) if not b]
        self.levels = np.delete(self.levels, idxs, axis=1)

        # Apply the gridding. Note we include a single point radial grid to
        # better interface with the function in analysemodel.

        self.xgrid = np.zeros(1)
        self.ygrid = self.estimate_grids(**kwargs)
        self.gridded = {}
        self.gridded['dens'] = self.grid_param(self.dens)
        self.gridded['gtemp'] = self.grid_param(self.gtemp)
        self.gridded['dtemp'] = self.grid_param(self.dtemp)
        self.gridded['abun'] = self.grid_param(self.abun)
        self.gridded['turb'] = self.grid_param(self.turb)
        self.gridded['levels'] = {}
        self.grid_levels(kwargs.get('nlevels', 5))

        return

    def grid_param(self, param):
        """
        Return a gridded version of param.
        """
        return griddata(self.zvals, param, self.ygrid, method='linear')

    def estimate_grids(self, **kwargs):
        """
        Return grids based on the points in the model.
        """
        npts = kwargs.get('npts', 500)
        return np.linspace(self.zvals.min(), self.zvals.max(), npts)

    def grid_levels(self, nlevels):
        """
        Grid the specified energy levels.
        """
        for j in np.arange(nlevels):
            if j in self.gridded['levels'].keys():
                continue
            self.gridded['levels'][j] = self.grid_param(self.levels[j])
        self.jmax = max(self.gridded['levels'].keys())
        if self.verbose:
            print('Gridded the first %d energy levels.' % (self.jmax))
            print('Use self.grid_levels() to read in more.\n')
        return
