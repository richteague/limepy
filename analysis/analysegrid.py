"""
Part of the limepy package.

Tools to read in and analyse the files produced with the `gridOutFile[3]`
parameter in LIME. Only considers a face on viewing geometry.

Functions to do:

    > Unit conversion to K.
    > Include velocity structure.
    > Tidy up the masking.
    > Include dust opacity.

"""

import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc
from limepy.analysis.collisionalrates import ratefile
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import warnings


class outputgrid:

    def __init__(self, grid=None, molecule=None, rates=None, **kwargs):
        """Read in and grid the LIME model."""

        if grid is None:
            raise ValueError('Must provide path to the grid (.ds).')
        self.path = grid
        self.filename = self.path.split('/')[-1]
        self.aux = os.path.dirname(os.path.realpath(__file__))
        self.aux = self.aux.replace('analysis', 'aux/')
        self.hdu = fits.open(self.path)
        self.verbose = kwargs.get('verbose', True)
        if not self.verbose:
            warnings.simplefilter("ignore")

        # -- Collisional Rates --
        #
        # The collisional rates are used to identify the frequencies
        # of the transitions and the statistical weights of the levels.
        # This currently doesn't support hyperfine components for, e.g., CN.
        # Can specify just a molecule or the direct path.

        if rates is None and molecule is None:
            raise ValueError('Must provide molecule or path to the rates.')
        elif molecule is not None:
            molecule = molecule.lower()
            rates = '%s.dat' % molecule
            if rates in os.listdir(self.aux):
                self.rates = ratefile(self.aux+rates)
            else:
                raise ValueError('Cannot find rates for %s.' % molecule)
        else:
            self.rates = ratefile(rates)

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
        # default of 10^4.

        self.dmask = self.dens > kwargs.get('min_density', 1e4)
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

        # -- Dust Properties --
        #
        # Read in the dust opacities. If none are given, use the standard
        # Jena opacities and load them as an array. They should be two columns
        # with the wavelength in [um] and the opacities in [cm^2/g]. Returns a
        # interpolation function which takes frequency in [Hz] and outputs the
        # opacity in the assumed units. By deafult, the gas-to-dust ratio
        # currently is set to 100 everywhere.

        self.opacities = kwargs.get('opacities', self.aux+'jena_thin_e6.tab')
        self.opacities = np.loadtxt(self.opacities).T
        if self.opacities.ndim != 2:
            raise ValueError("Can't read in dust opacities.")
        self.opacities = interp1d(sc.c * 1e6 / self.opacities[0][::-1],
                                  self.opacities[1][::-1], bounds_error=False,
                                  fill_value="extrapolated")
        self.g2d = kwargs.get('g2d', 100.)
        if self.verbose:
            print('Successfully attached dust opacities. Values loaded for')
            print('%.2f to %.2f GHz.' % (self.opacities.x.min()/1e9,
                  self.opacities.x.max()/1e9))
            print('Outside this opacities will be extrapolated.')
        return

    # -- General use functions. --

    def grid_levels(self, nlevels):
        """Grid the specified energy levels."""
        for j in np.arange(nlevels):
            if j in self.gridded['levels'].keys():
                continue
            self.gridded['levels'][j] = self.grid_param(self.levels[j],
                                                        self.method)
        self.jmax = max(self.gridded['levels'].keys())
        if self.verbose:
            print('Gridded the first %d energy levels.' % (self.jmax+1))
            print('Use self.grid_levels() to read in more.\n')
        return

    def grid_param(self, param, method='linear'):
        """Return a gridded version of param."""
        return griddata((np.hypot(self.xvals, self.yvals), self.zvals),
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

    # -- Physical Properties --
    #
    # Simple physical properties of the disk. These should be equal to what the
    # input values were for the LIME modelling.

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
        """Number density of molecules in selected level [/ccm]."""
        if level not in self.gridded['levels'].keys():
            self.grid_levels(level+1)
        nmol = self.gridded['dens'] * self.gridded['abun'] / 1e6
        return nmol * self.gridded['levels'][level]

    # -- Contribution Functions --
    #
    # Use cell_intensity(level) to call the cell intensities for each cell.
    # We can also calculate the emission observed from each cell with the
    # cell_emission function, or return a normalised and clipped version of
    # this with cell_contribution. All three functions have the option to
    # select the source from ['line', 'dust', 'both'].

    def cell_intensity(self, level, source='both'):
        """Unattenuated intensity [Jy/sr] of each cell."""
        if source == 'line':
            return self._line_intensity(level)
        elif source == 'dust':
            return self._dust_intensity(level)
        else:
            return self._both_intensity(level)

    def cell_emission(self, level, source='both'):
        """Intensity [Jy/sr] from each cell attenuated to disk surface."""
        cellint = self.cell_intensity(level, source=source)
        contrib = cellint * np.exp(-self.tau_cumulative(level, source=source))
        return np.where(np.isfinite(contrib), contrib, 0.0)

    def cell_contribution(self, level, source='both', mincont=1e-10):
        """Normalised cell contribution to the observed emission."""
        contrib = self.cell_emission(level, source=source)
        contrib = contrib / np.nansum(contrib, axis=0)
        return np.where(contrib < mincont, mincont, contrib)

    def _both_intensity(self, level):
        """Cell intensity [Jy/sr] for both line and dust emission."""
        I = 1e23 * self.S_both(level) * (1. - np.exp(-self.tau_both(level)))
        return np.where(np.isfinite(I), I, 0.0)

    def _line_intensity(self, level):
        """Returns the cell intensity [Jy/sr] only considering the line."""
        I = 1e23 * self.S_line(level) * (1. - np.exp(-self.tau_line(level)))
        return np.where(np.isfinite(I), I, 0.0)

    def _dust_intensity(self, level):
        """Returns the cell intensity [Jy/sr] only considering the dust."""
        I = 1e23 * self.S_dust(level) * (1. - np.exp(-self.tau_dust(level)))
        return np.where(np.isfinite(I), I, 0.0)
        return

    def alpha_line(self, level):
        """Line absorption coefficient [/cm]."""
        a = 1e4 * sc.c**2 / 8. / np.pi / self.rates.freq[level]**2
        a *= self.rates.EinsteinA[level] * self.phi(level)
        b = self.rates.g[level+1] / self.rates.g[level]
        b *= self.levelpop(level)
        b -= self.levelpop(level+1)
        a *= b
        return np.where(np.isfinite(a), a, 0.0)

    def alpha_dust(self, level):
        """Dust absorption coefficient [/cm]."""
        rho = self.gridded['dens'] / 1e6 / self.g2d
        kappa = self.opacities(self.rates.freq[level])
        alpha = rho * kappa
        return np.where(np.isfinite(alpha), alpha, 0.0)

    def emiss_line(self, level):
        """Line emissivity coefficient."""
        return self.S_line(level) * self.alpha_line(level)

    def emiss_dust(self, level):
        """Dust emissivity coefficient."""
        return self.S_dust(level) * self.alpha_dust(level)

    def S_dust(self, level):
        """Source function for the dust."""
        nu = self.rates.freq[level]
        B = 2. * sc.h * nu**3 / sc.c**2
        B /= np.exp(sc.h * nu / sc.k / self.gridded['dtemp']) - 1.
        return np.where(np.isfinite(B), B, 0.0)

    def S_line(self, level):
        """Source function for the line."""
        s = 2. * sc.h * self.rates.freq[level]**3 / sc.c**2
        ss = self.rates.g[level+1] / self.rates.g[level]
        ss *= self.levelpop(level) / self.levelpop(level+1)
        s /= (ss - 1.)
        return np.where(np.isfinite(s), s, 0.0)

    def S_both(self, level):
        """Source function for both line and dust."""
        source = self.alpha_dust(level) + self.alpha_line(level)
        source /= self.emiss_dust(level) + self.emiss_line(level)
        return np.where(np.isfinite(source), source, 0.0)

    def tau_line(self, level):
        """Optical depth of the line emission for each cell."""
        return self.alpha_line(level) * self.cellsize(self.ygrid)[:, None]

    def tau_dust(self, level):
        """Optical depth of the dust emission for each cell."""
        return self.alpha_dust(level) * self.cellsize(self.ygrid)[:, None]

    def tau_both(self, level):
        """Total optical depth of each cell."""
        return self.tau_line(level) + self.tau_dust(level)

    def tau_cumulative(self, level, source='both'):
        """Cumulative optical depth."""
        if source == 'line':
            tau = self.tau_line(level)
        elif source == 'dust':
            tau = self.tau_dust(level)
        else:
            tau = self.tau_both(level)
        return np.cumsum(tau[::-1], axis=0)[::-1]

    def radial_intensity(self, level, pixscale=None):
        """Radial intensity profile [Jy/sr]."""
        I = np.nansum(self.cell_emission(level), axis=0)
        return I

    def arcsec2sr(self, pixscale):
        """Convert a scale in arcseconds to a steradian."""
        return np.power(pixscale, 2.) * 2.35e-11

    def phi(self, level, offset=0.0):
        """Line fraction at line centre [/Hz]."""
        dnu = self.linewidth * self.rates.freq[level] / sc.c
        return 1. / dnu / np.sqrt(2. * np.pi)

    def normgauss(self, x, dx, x0=0.0):
        """Normalised Gaussian function."""
        func = np.exp(-0.5 * np.power((x-x0) / dx, 2.))
        func /= dx * np.sqrt(2. * np.pi)
        return np.where(np.isfinite(func), func, 0.0)

    # -- Weighted Properties --
    #
    # Use these functions to look at the abundance or flux weighted properties
    # of the disk model. The parameter must be a gridded property (that is, in
    # self.gridded.keys()) and the result can be returned as the [16, 50, 84]
    # percentiles or as [50, 50-16, 84-50] for error bar plotting.

    def fluxweighted(self, param, level, **kwargs):
        """Flux weighted percentiles of physical property."""
        if param not in self.gridded.keys():
            raise ValueError('Not valid parameter.')
        s = kwargs.get('source', 'both')
        f = self.cell_contribution(level, source=s)
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

    def excitiationtemperature(self, level):
        """Two level excitation temperature [K]."""
        T = self.levelpop(level) * self.rates.g[level+1]
        T /= self.levelpop(level+1) * self.rates.g[level]
        T = sc.h * self.rates.freq[level] / sc.k / np.log(T)
        return np.where(np.isfinite(T), T, 0.0)

    def emissionlayer(self, level, **kwargs):
        """Percentiles for the dominant emission layer [au]."""
        s = kwargs.get('source', 'both')
        f = self.cell_contribution(level, source=s)
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

    def machnumber(self, level):
        """Flux weighted Mach number of the turbulence."""
        temp = self.fluxweighted('gtemp', level)
        turb = self.fluxweighted('turb', level)
        cs = np.sqrt(sc.k * temp[0] / 2.35 / sc.m_p)
        return turb / cs

    def opticaldepth(self, level, **kwargs):
        """Flux weighted optical depth."""
        v = np.cumsum(self.tau(level)[::-1], axis=0)[::-1]
        f = self.cell_contribution(level)
        p = np.array([self.wpercentiles(v[:, i], f[:, i])
                      for i in xrange(self.xgrid.size)])
        if kwargs.get('percentiles', False):
            return p.T
        return self.percentilestoerrors(p.T)

    # -- Static Methods --
    #
    # Simple functions to aid with the calculations.

    @staticmethod
    def wpercentiles(data, weights, percentiles=[0.16, 0.5, 0.84], **kwargs):
        '''Weighted percentiles.'''
        if kwargs.get('onlypos', True):
            data = abs(data)
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
