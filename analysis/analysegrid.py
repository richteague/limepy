"""
Part of the limepy package.

Tools to read in and analyse the files produced with the `gridOutFile[3]`
parameter in LIME. Only considers a face on viewing geometry.

Functions to do:

    > Unit conversion to K.
    > Tidy up the masking.
    > Allow for a gas to dust function to be included rather than taking a
      disk-wide value, e.g. the default 100.

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

    def __init__(self, grid=None, rates=None, **kwargs):
        """Read in and grid the LIME model."""

        if grid is None:
            raise ValueError('Must provide path to the grid (.ds).')
        self.path = grid
        self.filename = self.path.split('/')[-1]
        self.aux = os.path.dirname(os.path.realpath(__file__))
        self.aux = self.aux.replace('analysis', 'aux/')
        self.hdu = fits.open(self.path)
        self.verbose = kwargs.get('verbose', True)
        self.depletion = kwargs.get('depletion', 1.0)

        if not self.verbose:
            warnings.simplefilter("ignore")

        # -- Collisional Rates --
        #
        # The collisional rates are used to identify the frequencies
        # of the transitions and the statistical weights of the levels.
        # This currently doesn't support hyperfine components for, e.g., CN.
        # Can specify just a molecule or the direct path.

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
        # default of 10^4. Include the depletion of the emitting molecule.

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
                                  fill_value="extrapolate")
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
            print('Gridded the first %d energy levels.' % (self.jmax))
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

    def columndensity(self, trans=None):
        """Returns the column density of the emitting molecule."""
        if trans is None:
            nmol = self.gridded['dens'] * self.gridded['abun'] / 1e6
        else:
            nmol = self.levelpop(trans)
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

    def linewidth_freq(self, trans):
        """Returns the frequency equivalent linewidth (stdev.) in [Hz]."""
        return self.linewidth * self.frequency(trans) / sc.c

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

    def cell_intensity(self, trans, **kwargs):
        """Unattenuated intensity [Jy/sr] of each cell."""
        source = kwargs.get('source', 'line')
        if source == 'line':
            return self._line_intensity(trans, **kwargs)
        elif source == 'dust':
            return self._dust_intensity(trans, **kwargs)
        else:
            return self._both_intensity(trans, **kwargs)

    def cell_emission(self, trans, **kwargs):
        """Intensity [Jy/sr] from each cell attenuated to disk surface."""
        cellint = self.cell_intensity(trans, **kwargs)
        contrib = cellint * np.exp(-self.tau_cumulative(trans, **kwargs))
        return np.where(np.isfinite(contrib), contrib, 0.0)

    def cell_contribution(self, trans, **kwargs):
        """Normalised cell contribution to the observed emission."""
        contrib = self.cell_emission(trans, **kwargs)
        contrib = contrib / np.nansum(contrib, axis=0)
        mincont = kwargs.get('mintcont', 1e-5)
        return np.where(contrib < mincont, mincont, contrib)

    def _both_intensity(self, trans, **kwargs):
        """Cell intensity [Jy/sr] for both line and dust emission."""
        I = 1e23 * self.S_both(trans, **kwargs)
        I *= (1. - np.exp(-self.tau_both(trans, **kwargs)))
        return np.where(np.isfinite(I), I, 0.0)

    def _line_intensity(self, trans, **kwargs):
        """Returns the cell intensity [Jy/sr] only considering the line."""
        I = 1e23 * self.S_line(trans, **kwargs)
        I *= (1. - np.exp(-self.tau_line(trans, **kwargs)))
        return np.where(np.isfinite(I), I, 0.0)

    def _dust_intensity(self, trans, **kwargs):
        """Returns the cell intensity [Jy/sr] only considering the dust."""
        I = 1e23 * self.S_dust(trans, **kwargs)
        I *= (1. - np.exp(-self.tau_dust(trans, **kwargs)))
        return np.where(np.isfinite(I), I, 0.0)
        return

    def alpha_line(self, trans, **kwargs):
        """Line absorption coefficient [/cm]."""
        n_u, n_l = self.leveldensities(trans)
        g_u, g_l = self.levelweights(trans)
        nu = self.frequency(trans)
        A = self.EinsteinA(trans)
        phi = self.phi(trans, **kwargs)
        a = 1.25e3 * sc.c**2 * A * n_u * phi / np.pi / nu**2
        a *= (n_l * g_u / n_u / g_l) - 1.
        return np.where(np.isfinite(a), a, 0.0)

    def alpha_dust(self, trans, **kwargs):
        """Dust absorption coefficient [/cm]."""
        g2d = kwargs.get('g2d', self.g2d)
        alpha = self.gridded['dens'] * sc.m_p / 1e6 / g2d
        alpha *= self.opacities(self.frequency(trans))
        return np.where(np.isfinite(alpha), alpha, 0.0)

    def emiss_line(self, trans, **kwargs):
        """Line emissivity coefficient."""
        return self.S_line(trans) * self.alpha_line(trans)

    def emiss_dust(self, trans, **kwargs):
        """Dust emissivity coefficient."""
        return self.S_dust(trans) * self.alpha_dust(trans)

    def S_dust(self, trans, **kwargs):
        """Source function for the dust."""
        nu = self.frequency(trans)
        B = 2. * sc.h * nu**3 / sc.c**2
        B /= np.exp(sc.h * nu / sc.k / self.gridded['dtemp']) - 1.
        return np.where(np.isfinite(B), B, 0.0)

    def S_line(self, trans, **kwargs):
        """Source function for the line."""
        n_u, n_l = self.leveldensities(trans)
        g_u, g_l = self.levelweights(trans)
        nu = self.frequency(trans)
        s = 2. * sc.h * nu**3 / sc.c**2 / (n_l * g_u / n_u / g_l - 1.)
        return np.where(np.isfinite(s), s, 0.0)

    def S_both_old(self, trans, **kwargs):
        """Source function for both line and dust."""
        source = self.alpha_dust(trans) + self.alpha_line(trans)
        source /= self.emiss_dust(trans) + self.emiss_line(trans)
        return np.where(np.isfinite(source), source, 0.0)

    def S_both(self, trans, **kwargs):
        """Source function for both line and dust."""
        source = self.emiss_dust(trans) + self.emiss_line(trans)
        source /= self.alpha_dust(trans) + self.alpha_line(trans)
        return np.where(np.isfinite(source), source, 0.0)

    def tau_line(self, trans, **kwargs):
        """Optical depth of the line emission for each cell."""
        tau = self.alpha_line(trans, **kwargs)
        return tau * self.cellsize(self.ygrid)[:, None]

    def tau_dust(self, trans, **kwargs):
        """Optical depth of the dust emission for each cell."""
        tau = self.alpha_dust(trans, **kwargs)
        return tau * self.cellsize(self.ygrid)[:, None]

    def tau_both(self, trans, **kwargs):
        """Total optical depth of each cell."""
        tau = self.tau_line(trans, **kwargs)
        return tau + self.tau_dust(trans, **kwargs)

    def tau_cumulative(self, trans, **kwargs):
        """Cumulative optical depth."""
        source = kwargs.get('source', 'both')
        if source == 'line':
            tau = self.tau_line(trans, **kwargs)
        elif source == 'dust':
            tau = self.tau_dust(trans, **kwargs)
        else:
            tau = self.tau_both(trans, **kwargs)
        return np.cumsum(tau[::-1], axis=0)[::-1]

    def radial_intensity(self, trans, pixscale=None):
        """Radial intensity profile [Jy/sr]."""
        I = np.nansum(self.cell_emission(trans), axis=0)
        return I

    def arcsec2sr(self, pixscale):
        """Convert a scale in arcseconds to a steradian."""
        return np.power(pixscale, 2.) * 2.35e-11

    def phi(self, trans, **kwargs):
        """Line profile [/Hz]."""
        wings = kwargs.get('wings', 0.0)
        dnu = self.linewidth_freq(trans)
        return np.exp(-wings) / dnu / np.sqrt(2. * np.pi)

    def normgauss(self, x, dx, x0=0.0):
        """Normalised Gaussian function."""
        func = np.exp(-1.0 * np.power((x-x0) / dx, 2.))
        func /= dx * np.sqrt(np.pi)
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
        f *= self.cellsize(self.ygrid)
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

    def excitiationtemperature(self, trans):
        """Two level excitation temperature [K]."""
        n_u, n_l = self.leveldensities(trans)
        g_u, g_l = self.levelweights(trans)
        nu = self.frequency(trans)
        T = sc.h * nu / sc.k / np.log(n_l * g_u / n_u / g_l)
        return np.where(np.isfinite(T), T, 0.0)

    def emissionlayer(self, trans, **kwargs):
        """Percentiles for the dominant emission layer [au]."""
        f = self.cell_contribution(trans, **kwargs)
        p = np.array([self.wpercentiles(abs(self.ygrid), f[:, i])
                      for i in xrange(self.xgrid.size)])
        if kwargs.get('percentiles', False):
            return p.T
        return self.percentilestoerrors(p.T)

    def molecularlayer(self, **kwargs):
        """Percentiles for the dominant emission layer [au]."""
        f = self.gridded['abun'] * self.gridded['dens']
        f *= self.cellsize(self.ygrid)
        p = np.array([self.wpercentiles(abs(self.ygrid), f[:, i])
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
        v = self.tau_cumulative(level, **kwargs)
        f = self.cell_contribution(level, **kwargs)
        p = np.array([self.wpercentiles(v[:, i], f[:, i])
                      for i in xrange(self.xgrid.size)])
        if kwargs.get('percentiles', False):
            return p.T
        return self.percentilestoerrors(p.T)

    def levelweights(self, trans):
        """Weights of the upper / lower energy levels of the transition."""
        return self.rates.levels[trans+2].g, self.rates.levels[trans+1].g

    def leveldensities(self, trans):
        """Returns the upper / lower energy number densities [/ccm]."""
        return self.levelpop(trans+1), self.levelpop(trans)

    def frequency(self, trans, **kwargs):
        """Returns the frequency [Hz] of the transition."""
        return self.rates.lines[trans+1].freq

    def EinsteinA(self, trans):
        """Returns the Einstein A coefficient [Hz] of the transition."""
        return self.rates.lines[trans+1].A

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
        """Converts [16, 50, 84] percentiles to [<x>, -dx, +dx]."""
        profile = np.ones(percentiles.shape)
        profile[0] = percentiles[1]
        profile[1] = percentiles[1] - percentiles[0]
        profile[2] = percentiles[2] - percentiles[1]
        return profile
