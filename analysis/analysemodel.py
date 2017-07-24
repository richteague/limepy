"""
Part of the limepy package.

Tools to read in and analyse the files produced with the `gridOutFile[3]`
parameter in LIME. Only considers a face on viewing geometry.

Functions to do:

    > Unit conversion to K.
    > Tidy up the masking.
    > Allow for a gas to dust function to be included rather than taking a
      disk-wide value, e.g. the default 100.
    > Include depletion.

"""

import numpy as np
import scipy.constants as sc
from collisionalrates import ratefile
from scipy.interpolate import interp1d
from models import diskmodel, slabmodel
import warnings


class outputgrid:

    def __init__(self, path, modeltype='disk', molecule=None, **kwargs):
        """Read in and grid the LIME model."""

        # Read in the LIME model.
        if modeltype == 'disk':
            self.model = diskmodel(path, **kwargs)
        elif modeltype == 'slab':
            self.model = slabmodel(path, **kwargs)
        else:
            raise ValueError("'modeltype' must be 'disk' or 'slab'.")

        # Read in the collisional rates.
        if molecule is not None:
            self.molecule = molecule
            self.rates = ratefile(molecule)
        else:
            raise ValueError("Must specify which molecule.")

        # How silent?
        self.verbose = self.model.verbose
        if not self.verbose:
            warnings.simplefilter("ignore")

        # Quickly retrieve the properties from the model.
        # TODO: Would it better to have these without quicknames?
        self.aux = self.model.aux
        self.xgrid = self.model.xgrid
        self.ygrid = self.model.ygrid
        self.gridded = self.model.gridded
        self.grid_levels = self.model.grid_levels

        # -- Dust Properties --
        #
        # TODO! Can we add this to a separate module?
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

    # -- Physical Properties --

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
        if self.xgrid.size > 1:
            return tau * self.cellsize(self.ygrid)[:, None]
        return tau * self.cellsize(self.ygrid)

    def tau_dust(self, trans, **kwargs):
        """Optical depth of the dust emission for each cell."""
        tau = self.alpha_dust(trans, **kwargs)
        if self.xgrid.size > 1:
            return tau * self.cellsize(self.ygrid)[:, None]
        return tau * self.cellsize(self.ygrid)

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
        return np.exp(-wings) / dnu / np.sqrt(np.pi)

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
        if self.xgrid.size > 1:
            p = np.array([self.wpercentiles(v[:, i], f[:, i])
                          for i in xrange(self.xgrid.size)])
        else:
            p = self.wpercentiles(v, f)
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
        if self.xgrid.size > 1:
            p = np.array([self.wpercentiles(abs(self.ygrid), f[:, i])
                          for i in xrange(self.xgrid.size)])
            if kwargs.get('percentiles', False):
                return p.T
            return self.percentilestoerrors(p.T)
        p = self.wpercentiles(self.ygrid, f)
        if kwargs.get('percentiles', False):
            return p
        return self.percentilestoerrors(p)

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
