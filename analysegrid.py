"""
Part of the limepy package.

Tools to read in and analyse the files produced with the `gridOutFile[3]`
parameter in LIME.
"""

import numpy as np
from astropy.io import fits
import scipy.constants as sc
from scipy.interpolate import griddata
from analyseLIME.readLAMDA import ratefile


class outputgrid:

    def __init__(self, path, rates=None, **kwargs):
        """Read in the grid FITS file."""
        self.path = path
        self.filename = self.path.split('/')[-1]
        self.hdu = fits.open(self.path)
        if rates is None:
            self.rates = rates
        else:
            self.rates = ratefile(rates)

        self.verbose = kwargs.get('verbose', True)
        if self.verbose:
            print(self.hdu.info())
            print('\n')
            if self.rates is None:
                print('Warning: no collisional rates provided!\n')

        # Currently only important are the grid [1] and level populations [4]
        # from the grid. Both [2] and [3] are for the Delanunay triangulation
        # and can thus be ignored.

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

        # Physical properties at each cell.
        # If dtemp == -1, then use gtemp.

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
        # TODO: Allow more than one abundance.

        self.nabun = len([n for n in self.names if 'ABUNMOL' in n])
        if self.nabun > 1:
            raise NotImplementedError()
        self.abun = self.grid.data['ABUNMOL1'][self.notsink]
        self.velo = np.array([self.grid.data['VEL%d' % i][self.notsink]
                              for i in [1, 2, 3]])
        self.turb = self.grid.data['TURBDPLR'][self.notsink]

        # Mask out all points with a total density of <= min_density, with a
        # default of 10^3.

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
        self.abun = self.abun[self.dmask]
        self.turb = self.turb[self.dmask]

        # Excitation properties. Remove all the sink particles.

        pops = self.hdu[4].data.T
        idxs = [i for i, b in enumerate(self.notsink) if not b]
        self.levels = np.delete(pops, idxs, axis=1)
        idxs = [i for i, b in enumerate(self.dmask) if not b]
        self.levels = np.delete(self.levels, idxs, axis=1)

        # Grid the data. If no axes are provided then estimate them.
        # It is onto these that the data is gridded and subsequent
        # calculations performed. Note that if log grids are used, it only
        # returns a grid on the (0, zmax] interval.

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
        self.npnts = self.xgrid.size
        self.log = kwargs.get('log', False)

        # With the grids, grid the parameters and store them in a dictionary.
        # Only read in a certain amount of energy levels to quick it quick.

        method = kwargs.get('method', 'linear')
        if self.verbose:
            print('Beginning gridding using %s interpolation.' % method)
            if kwargs.get('log', False):
                print('Axes are logarithmically spaced.')
            if method == 'nearest':
                print('Warning: neartest may produce unwanted features.')
        self.gridded = {}
        self.gridded['dens'] = self.grid_param(self.dens, method)
        self.gridded['gtemp'] = self.grid_param(self.gtemp, method)
        self.gridded['dtemp'] = self.grid_param(self.dtemp, method)
        self.gridded['abun'] = self.grid_param(self.abun, method)
        self.gridded['turb'] = self.grid_param(self.turb, method)

        self.jmax = kwargs.get('nlevels', 5)
        if self.verbose:
            print('Gridding the first %d energy levels.\n' % self.jmax)
        self.gridded['levels'] = {j: self.grid_param(self.levels[j], method)
                                  for j in np.arange(self.jmax)}

        return

    def grid_param(self, param, method='linear'):
        """Return a gridded version of param."""
        return griddata((np.hypot(self.xvals, self.yvals), self.zvals),
                        param, (self.xgrid[None, :], self.ygrid[:, None]),
                        method=method, fill_value=0.0, rescale=True)

    def estimate_grids(self, **kwargs):
        """Return grids based on points."""
        npts = kwargs.get('npts', 500)
        assert type(npts) == int
        xmin = self.rvals.min()
        xmax = self.rvals.max() * 1.05
        ymin = abs(self.zvals).min()
        ymax = abs(self.zvals).max() * 1.05
        if kwargs.get('log', False):
            xgrid = np.logspace(np.log10(xmin), np.log10(xmax), npts)
            ygrid = np.logspace(np.log10(ymin), np.log10(ymax), npts)
            return xgrid, ygrid
        xgrid = np.linspace(xmin, xmax, npts)
        ygrid = np.linspace(-ymax, ymax, 5*npts)
        return xgrid, ygrid

    @property
    def columndensity(self):
        """Returns the column density of the emitting molecule."""
        nmol = self.gridded['dens'] * self.gridded['abun'] / 1e6
        if self.ygrid.min() < 0:
            return np.trapz(nmol, x=self.ygrid*sc.au*1e2, axis=0)
        return 2. * np.trapz(nmol, x=self.ygrid*sc.au*1e2, axis=0)

    @property
    def surfacedensity(self):
        """Return the surface density of the main collider."""
        nh2 = self.gridded['dens'] / 1e6
        if self.ygrid.min() < 0:
            return np.trapz(nh2, x=self.ygrid*sc.au*1e2, axis=0)
        return 2. * np.trapz(nh2, x=self.ygrid*sc.au*1e2, axis=0)

    @property
    def linewidth(self):
        """Returns the local total linewidth (stdev.) in [m/s]."""
        return np.hypot(self.gridded['turb'], self.thermalwidth)

    @property
    def thermalwidth(self):
        """Returns the local thermal width in [m/s]."""
        dV = 2. * sc.k * self.gridded['gtemp'] / 2. / self.rates.mu / sc.m_p
        return np.sqrt(dV)

    def levelpop(self, level):
        """Number density of molecules in required level [/ccm]."""
        nmol = self.gridded['dens'] * self.gridded['abun'] / 1e6
        return nmol * self.gridded['levels'][level]

    def anu(self, level):
        """Absorption coefficient [/cm]."""
        a = 1e4 * sc.c**2 / 8. / np.pi / self.rates.freq[level]**2
        a *= self.rates.EinsteinA[level]
        a *= self.phi(level)
        b = self.rates.g[level+1] / self.rates.g[level]
        b *= self.levelpop(level)
        b -= self.levelpop(level+1)
        a *= b
        return np.where(np.isfinite(a), a, 0.0)

    def jnu(self, level):
        """Emission coefficient [erg / s / ccm / Hz / sr]."""
        j = 6.62e-27 * self.phi(level) * self.rates.freq[level] / 4. / np.pi
        j *= self.levelpop(level+1) * self.rates.EinsteinA[level]
        return np.where(np.isfinite(j), j, 0.0)

    def Snu(self, level):
        """Source function."""
        s = self.jnu(level) / self.anu(level)
        return np.where(np.isfinite(s), s, 0.0)

    def tau(self, level):
        """Optical depth of each cell.."""
        return self.anu(level) * self.cellsize(self.ygrid)[:, None]

    def cell_intensity(self, level, pixscale=None):
        """Intensity from each cell in [Jy/sr] or [Jy/pix]."""
        I = 1e23 * self.Snu(level) * (1. - np.exp(-self.tau(level)))
        I = np.where(np.isfinite(I), I, 0.0)
        if pixscale is None:
            return I
        return I * self.arcsec2sr(pixscale)

    def cell_emission(self, level, pixscale=None):
        """Intensity from each cell attenuated to disk surface [Jy/area]."""
        cellint = self.cell_intensity(level, pixscale)
        cumtaup = np.cumsum(self.tau(level)[::-1], axis=0)[::-1]
        contrib = cellint * np.exp(-cumtaup)
        contrib = np.where(np.isfinite(contrib), contrib, 0.0)
        return contrib

    def cell_contribution(self, level, pixscale=None, **kwargs):
        """Normalised cell contribution to the intensity."""
        contrib = self.cell_emission(level, pixscale)
        contrib = contrib / np.nansum(contrib, axis=0)
        mincont = kwargs.get('mincont', 1e-10)
        return np.where(contrib < mincont, 1e-10, contrib)

    def radial_intensity(self, level, pixscale=None):
        """Radial intensity profile [Jy/sr]."""
        return np.nansum(self.cell_emission(level, pixscale), axis=0)

    def arcsec2sr(self, pixscale):
        """Convert a scale in arcseconds to a steradian."""
        return np.power(pixscale, 2.) * 2.35e-11

    def phi(self, level):
        """Line fraction at line centre [/Hz]."""
        dnu = self.linewidth * self.rates.freq[level] / sc.c
        return 1. / dnu / np.sqrt(2. * np.pi)

    def normgauss(self, x, dx, x0=0.0):
        """Normalised Gaussian function."""
        func = np.exp(-0.5 * np.power((x-x0) / dx, 2.))
        func /= dx * np.sqrt(2. * np.pi)
        return np.where(np.isfinite(func), func, 0.0)

    def fluxweighted(self, param, level, **kwargs):
        """Flux weighted percentiles of physical property."""
        if param not in self.gridded.keys():
            raise ValueError('Not valid parameter.')
        f = self.cell_contribution(level)
        v = self.gridded[param]
        p = np.array([self.wpercentiles(v[:, i], f[:, i])
                      for i in xrange(self.xgrid.size)])
        if kwargs.get('percentiles', False):
            return p.T
        return self.percentilestoerrors(p.T)

    def abundanceweighted(self, param, **kwargs):
        """Abundance weighted percentile of physical property."""
        if param not in self.gridded.keys():
            raise ValueError('Not valid parameter.')
        f = self.gridded['abun'] * self.gridded['dens']
        v = self.gridded[param]
        p = np.array([self.wpercentiles(v[:, i], f[:, i])
                      for i in xrange(self.xgrid.size)])
        if kwargs.get('percentiles', False):
            return p.T
        return self.percentilestoerrors(p.T)

    def cellsize(self, axis, unit='cm'):
        """Returns the cell sizes."""
        mx = axis.size
        dx = np.diff(axis)
        ss = [(dx[max(0, i-1)]+dx[min(i, mx-2)])*0.5 for i in range(mx)]
        if unit == 'cm':
            return np.squeeze(ss) * 1e2 * sc.au
        elif unit == 'au':
            return np.squeeze(ss)
        else:
            raise ValueError("unit must be 'au' or 'cm'.")

    def exictiationtemperature(self, level):
        """Two level excitation temperature [K]."""
        T = self.levelpop(level) * self.rates.g[level+1]
        T /= self.levelpop(level+1) * self.rates.g[level]
        T = sc.h * self.rates.freq[level] / sc.k / np.log(T)
        return np.where(np.isfinite(T), T, 0.0)

    def emissionlayer(self, level, **kwargs):
        """Percentiles for the dominant emission layer [au]."""
        f = self.cell_contribution(level)
        p = np.array([self.wpercentiles(self.ygrid, f[:, i])
                      for i in xrange(self.xgrid.size)])
        if kwargs.get('percentiles', False):
            return p.T
        return self.percentilestoerrors(p.T)

    def molecularlayer(self, **kwargs):
        """Percentiles for the dominant emission layer [au]."""
        f = self.gridded['abun'] * self.gridded['dens']
        p = np.array([self.wpercentiles(self.ygrid, f[:, i])
                      for i in xrange(self.xgrid.size)])
        if kwargs.get('percentiles', False):
            return p.T
        return self.percentilestoerrors(p.T)

    @staticmethod
    def wpercentiles(data, weights, percentiles=[0.16, 0.5, 0.84]):
        '''Weighted percentiles.'''
        idx = np.argsort(data)
        sorted_data = np.take(data, idx)
        sorted_weights = np.take(weights, idx)
        cum_weights = np.add.accumulate(sorted_weights)
        scaled_weights = (cum_weights - 0.5 * sorted_weights) / cum_weights[-1]
        spots = np.searchsorted(scaled_weights, percentiles)
        wp = []
        for s, p in zip(spots, percentiles):
            if s == 0:
                wp.append(sorted_data[s])
            elif s == data.size:
                wp.append(sorted_data[s-1])
            else:
                f1 = (scaled_weights[s] - p)
                f1 /= (scaled_weights[s] - scaled_weights[s-1])
                f2 = (p - scaled_weights[s-1])
                f2 /= (scaled_weights[s] - scaled_weights[s-1])
                wp.append(sorted_data[s-1] * f1 + sorted_data[s] * f2)
        return np.array(wp)

    @staticmethod
    def percentilestoerrors(percentiles):
        """Converts [16,50,84] percentiles to <x> +/- dx."""
        profile = np.ones(percentiles.shape)
        profile[0] = percentiles[1]
        profile[1] = percentiles[1] - percentiles[0]
        profile[2] = percentiles[2] - percentiles[1]
        return profile
