"""
Part of the limepy package.

Class to help read in the models from LIME. Basic functions to return the
moments of the data and radial profiles. Applies simple continuum subtraction.

Functions to do:

    > Strip excess continuum channels.
    > Convert between Jy/pix and K.
    > Second moments.

"""

import numpy as np
from astropy.io import fits
import scipy.constants as sc
from scipy.interpolate import interp1d


class cube:

    def __init__(self, path, **kwargs):

        self.filename = path

        # Read in the cube and the appropriate axes. Currently assume that the
        # position axes are the same.

        self.data = np.squeeze(fits.getdata(self.filename, 0))
        self.posax = self.readpositionaxis()
        self.npix = self.posax.size
        self.dpix = np.diff(self.posax).mean()
        if not kwargs.get('restfreq', False):
            try:
                self.nu = fits.getval(self.filename, 'restfreq', 0)
            except KeyError:
                self.nu = fits.getval(self.filename, 'restfrq', 0)
        else:
            self.nu = kwargs.get('restfreq')
        if fits.getval(self.filename, 'ctype3').lower() == 'freq':
            self.specax = self.readspectralaxis()
            self.velax = (self.nu - self.specax) * sc.c / self.nu / 1e3
        else:
            self.velax = self.readvelocityaxis()
            self.specax = self.velax * 1e3 * self.nu / sc.c
        self.nchan = self.velax.size
        self.unit = fits.getval(self.filename, 'bunit').lower()

        # Parse the information about the geometry of the model from the
        # header. If not available, use the **kwargs to fill in. Should warn
        # if a value is not found and use np.nan as a filler.

        self.inc, self.pa, self.dist, self.azi = self.readheader(**kwargs)
        self.projax = self.posax * self.dist
        self.mstar = self.readvalue('mstar', fillval=1.0)

        # If the trio of (inc, pa, azi) are all defined, then we can calculate
        # the projected positions of each pixel assuming a thin disk.

        self.rvals, self.tvals = self.deprojectedcoords(**kwargs)

        # Remove the continuum for a quick analysis of the line emission.
        # Make a quick check that the first two and last two channels are the
        # same, otherwise raise a warning suggesting that continuum subtraction
        # may be slightly wrong.

        self.cont, self.line = self.subtractcontinuum(**kwargs)
        self.totalintensity = np.nansum(self.zerothmoment())

        # Attempt to read beam properties from the header. If none are found,
        # take values as a pixel size. This allows a quick conversion between
        # K and Jy/pix or Jy/beam.

        self.bmin, self.bmaj, self.bpa = self.readbeam(**kwargs)

        return

    def deprojectedcoords(self, **kwargs):
        """Returns the deprojected (r, theta) coordinates."""

        # First check that the inclination and position angle are specified.

        if np.isnan(self.inc):
            print('No inclination specified. Assuming 0 rad.')
            self.inc = 0.0
        if np.isnan(self.pa):
            print('No position angle specified. Assuming 0 rad.')
            self.pa = 0.0

        # This is a two step process. First, derotate everything back to a
        # position angle of 0.0 (i.e. the major axis aligned with the x-axis),
        # then deproject along the y-axis.

        x_obs = self.posax[None, :] * np.ones(self.npix)[:, None]
        y_obs = self.posax[:, None] * np.ones(self.npix)[None, :]
        x_rot = x_obs * np.cos(self.pa) - y_obs * np.sin(self.pa)
        y_rot = x_obs * np.sin(self.pa) + y_obs * np.cos(self.pa)
        x_dep = x_rot
        y_dep = y_rot / np.cos(self.inc)
        return np.hypot(y_dep, x_dep), np.arctan2(y_dep, x_dep)

    def keplerian(self, **kwargs):
        """Projected Keplerian velocity at the given pixel."""
        mstar = kwargs.get('mstar', self.mstar)
        keplerian = np.sqrt(sc.G * mstar * 1.989e30 / self.rvals / sc.au)
        return keplerian * np.sin(self.inc) * np.cos(self.tvals)

    def averagespectra(self, bins=None, nbins=None, **kwargs):
        """Azimuthally average spectra."""
        shifted = self.shiftspectra(**kwargs)
        shifted = shifted.reshape((shifted.shape[0], -1)).T
        bins = self.radialbins(bins=bins, nbins=nbins)
        ridxs = np.digitize(self.rvals.ravel(), bins)
        rpnts = np.mean([bins[1:], bins[:-1]], axis=0)
        lines = [np.average(shifted[ridxs == r], axis=0)
                 for r in range(1, bins.size)]
        scatter = [np.std(shifted[ridxs == r], axis=0)
                   for r in range(1, bins.size)]
        return rpnts, np.squeeze(lines), np.squeeze(scatter)

    def shiftspectra(self, **kwargs):
        """Shift all pixels by the Keplerian offset."""
        if self.inc == 0.0:
            return self.line
        shifted = np.zeros(self.line.shape)
        offset = self.keplerian(**kwargs)
        rout = kwargs.get('rout', 2.0)
        for i in range(self.posax.size):
            for j in range(self.posax.size):
                if self.rvals > rout:
                    continue
                shift = interp1d(self.velax - offset[j, i], self.line[:, j, i],
                                 fill_value=0.0, bounds_error=False,
                                 assume_sorted=True)
                shifted[:, j, i] = shift(self.velax)
        return shifted

    def zerothmoment(self, **kwargs):
        """Returns the zeroth moment of the data."""
        if kwargs.get('removecont', True):
            return np.trapz(self.line, self.velax, axis=0)
        return np.trapz(self.data, self.velax, axis=0)

    def firstmoment(self, method='weighted', **kwargs):
        """Returns the first moment of the data."""
        if not (method in 'weighted' or method in 'maximum'):
            raise ValueError("method must be 'weighted' or 'maximum'.")
        if method in 'weighted':
            return self.weightedfirst(**kwargs)
        return self.maximumfirst(**kwargs)

    def secondmoment(self, **kwargs):
        """Returns the second moment of the data."""
        raise NotImplementedError()
        return

    def maximumintensity(self, **kwargs):
        """Returns the peak emission along each pixel."""
        if kwargs.get('removecont', True):
            return np.amax(self.line, axis=0)
        return np.amax(self.data, axis=0)

    def percentilestoerrors(self, percentiles):
        """Converts [16,50,84] percentiles to <x> +/- dx."""
        profile = np.ones(percentiles.shape)
        profile[0] = percentiles[1]
        profile[1] = percentiles[1] - percentiles[0]
        profile[2] = percentiles[2] - percentiles[1]
        return profile

    def maximumprofile(self, bins=None, nbins=None, **kwargs):
        """Azimuthally averaged maximum flux density profile."""
        bins = self.radialbins(bins=bins, nbins=nbins)
        zeroth = self.maximumintensity(**kwargs).ravel()
        ridxs = np.digitize(self.rvals.ravel(), bins)
        pvals = kwargs.get('percentiles', [0.16, 0.5, 0.94])
        rpnts = np.mean([bins[1:], bins[:-1]], axis=0)
        percentiles = [np.percentile(zeroth[ridxs == r], pvals)
                       for r in range(1, bins.size)]
        percentiles = np.squeeze(percentiles).T
        if kwargs.get('uncertainty', True):
            return rpnts, self.percentilestoerrors(percentiles)
        return rpnts, percentiles

    def intensityprofile(self, nbins=None, bins=None, **kwargs):
        """Returns the azimutahlly averaged intensity profile."""
        bins = self.radialbins(bins=bins, nbins=nbins)
        zeroth = self.zerothmoment(**kwargs).ravel()
        ridxs = np.digitize(self.rvals.ravel(), bins)
        pvals = kwargs.get('percentiles', [16, 50, 84])
        rpnts = np.mean([bins[1:], bins[:-1]], axis=0)
        percentiles = [np.percentile(zeroth[ridxs == r], pvals)
                       for r in range(1, bins.size)]
        percentiles = np.squeeze(percentiles).T
        if kwargs.get('uncertainty', True):
            return rpnts, self.percentilestoerrors(percentiles)
        return rpnts, percentiles

    def radialbins(self, nbins=None, bins=None, **kwargs):
        """Returns the radial bins."""
        if bins is None and nbins is None:
            raise ValueError("Specify either 'bins' or 'nbins'.")
        if bins is not None and nbins is not None:
            raise ValueError("Specify either 'bins' or 'nbins'.")
        if bins is None:
            return np.linspace(0., self.posax.max(), nbins+1)
        return bins

    def weightedfirst(self, **kwargs):
        """First moment from intsensity weighted average."""
        emission = np.sum(self.line, axis=0)
        emission = np.where(emission != 0, 1, 0)
        emission = emission * np.ones(self.line.shape)
        noise = 1e-30 * np.random.random(self.line.shape)
        weights = np.where(emission, self.line, noise)
        vcube = self.velax[:, None, None] * np.ones(weights.shape)
        first = np.average(vcube, weights=weights, axis=0)
        emission = np.sum(emission, axis=0)
        return np.where(emission, first, kwargs.get('mask', np.nan))

    def maximumfirst(self, **kwargs):
        """First moment from the maximum value."""
        vidx = np.argmax(self.line, axis=0)
        vmax = np.take(self.velax, vidx)
        return np.where(vidx != 0, vmax, np.nan)

    def subtractcontinuum(self, **kwargs):
        """Return the line-only data."""
        cchan = kwargs.get('cont_chan', 0)
        cont = self.data[cchan].copy()
        if not all([np.isclose(np.sum(cont - self.data[i]), 0)
                    for i in [1, -1, -2]]):
            print('Emission in edge channels.')
        line = np.array([chan - cont for chan in self.data])
        return cont, line

    def readvelocityaxis(self):
        """Return velocity axis in [km/s]."""
        a_len = fits.getval(self.filename, 'naxis3')
        a_del = fits.getval(self.filename, 'cdelt3')
        a_pix = fits.getval(self.filename, 'crpix3')
        a_ref = fits.getval(self.filename, 'crval3')
        return (a_ref + (np.arange(a_len) - a_pix + 1) * a_del) / 1e3

    def readspectralaxis(self):
        """Returns the spectral axis in [Hz]."""
        a_len = fits.getval(self.filename, 'naxis3')
        a_del = fits.getval(self.filename, 'cdelt3')
        a_pix = fits.getval(self.filename, 'crpix3')
        a_ref = fits.getval(self.filename, 'crval3')
        return a_ref + (np.arange(a_len) - a_pix + 1) * a_del

    def readpositionaxis(self):
        """Returns the position axis in ["]."""
        a_len = fits.getval(self.filename, 'naxis2')
        a_del = fits.getval(self.filename, 'cdelt2')
        a_pix = fits.getval(self.filename, 'crpix2')
        return 3600. * ((np.arange(1, a_len+1) - a_pix) * a_del)

    def readheader(self, **kwargs):
        """Reads the model properties."""
        inc = self.readvalue('inc', **kwargs)
        pa = self.readvalue('pa', **kwargs)
        dist = self.readvalue('distance', **kwargs)
        azi = self.readvalue('azi', **kwargs)
        return inc, pa, dist, azi

    def readbeam(self, **kwargs):
        """Reads in the beam properties [rad]."""
        bmin = self.readvalue('bmin', fillval=self.dpix/3600., **kwargs)
        bmaj = self.readvalue('bmaj', fillval=self.dpix/3600., **kwargs)
        bpa = self.readvalue('bpa', fillval=0.0, **kwargs)
        return np.radians(bmin), np.radians(bmaj), np.radians(bpa)

    def readvalue(self, key, fillval=np.nan, **kwargs):
        """Attempt to read a value from the header."""
        assert type(key) is str
        try:
            value = fits.getval(self.filename, key, 0)
        except:
            value = kwargs.get(key, fillval)
        return value

    @property
    def spectrum(self):
        """Integrated intensity [Jy]."""
        tosum = self.line.copy()
        if self.unit == 'jy/beam':
            bmin = 3600. * np.degrees(self.bmin)
            bmaj = 3600. * np.degrees(self.bmaj)
            tosum *= self.dpix**2 * 4. * np.log(2.) / bmin / bmaj / np.pi
        elif self.unit == 'k':
            raise NotImplementedError()
        return np.squeeze([np.sum(c) for c in tosum])

    @property
    def Tmb(self):
        """Conversion to brightness temperature."""
        if self.unit.lower() == 'k':
            return 1.
        T = 2. * np.log(2.) * sc.c**2 / sc.k / self.nu**2. * 1e-26
        return T / np.pi / self.bmin / self.bmaj
