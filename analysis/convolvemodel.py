"""
Take a LIME model and then convolve it with a beam.

    -- Input --

    path:               Path the cube you wish to convolve with a beam.
    bmaj:               FWHM of the beam major axis [arcsec].
    bmin [optional]:    FWHM of the beam minor axis in [arcsec]. If none is
                        specified, assume a circular beam.
    bpa [optional]:     Position angle of the beam defined as the angle between
                        North and the major axis in an anticlockwise direction
                        in [degrees].
    fast [optional]:    If True, use FFT over normal convolution.
    output [optional]:  The name of the output files. If none is specified,
                        append the filename with bmin_bmaj_bpa.fits.
    hanning [optional]: If set, will Hanning smooth the spectral dimension of
                        the data assuming the natural channel width of ALMA
                        data is 15 kHz.
    dcorr [optional]:   Width of the Hanning smoothing kernel to use.
    chan [semi-opt]:    Unit of the width of the correlator kernel.

"""

import numpy as np
from astropy.io import fits
from astropy.convolution import Kernel, convolve, convolve_fft
from scipy.interpolate import interp1d
import scipy.constants as sc


def convolvecube(path, bmaj, bmin=None, bpa=0.0, hanning=True,
                 fast=True, output=None, dcorr=2., unit='chan'):
    """Convolve a LIME model with a 2D Gaussian beam."""
    fn = path.split('/')[-1]
    dir = '/'.join(path.split('/')[:-1])+'/'
    if dir[0] == '/':
        dir = dir[1:]
    data = fits.getdata(path)

    # Return the pixel scaling in [arcsec / pix].
    dpix = np.mean(np.diff(readpositionaxis(path)))

    # Make sure that the minimum and maximum values are in the correct order.
    if bmin is None:
        bmin = bmaj
    if bmin > bmaj:
        temp = bmaj
        bmaj = bmin
        bmin = temp

    # Calculate the beam kernel and convolve the cube. Two tests to see if it
    # is worth while running the convolution: (a) is there a beam specified and
    # (b) is the resulting kernel larger than a pixel?

    if bmaj > 0.0:
        beam = beamkernel(bmaj, bmin, bpa, dpix)
        if beam.array.size > 1:
            print("Beginning beam convolution...")
            if fast:
                ccube = np.array([convolve_fft(c, beam) for c in data])
            else:
                ccube = np.array([convolve(c, beam) for c in data])
        else:
            ccube = data
    else:
        ccube = data

    # Apply Hanning smoothing by default. TODO: Is there a way to speed this
    # up rather than looping through each pixel?
    if hanning:
        kernel = hanningkernel(path, dcorr=dcorr, unit=unit)
        if kernel.array.size > 1:
            print("Beginning Hanning smoothing...")
            for i in range(ccube.shape[2]):
                for j in range(ccube.shape[1]):
                    if fast:
                        ccube[:, j, i] = convolve_fft(ccube[:, j, i], kernel)
                    else:
                        ccube[:, j, i] = convolve(ccube[:, j, i], kernel)

    # Use the header of the input cube as the basis for the new cube.
    # Add in additional header keywords describing the beam.
    hdu = fits.open(path)
    hdr = hdu[0].header
    if hanning:
        hdr['HANNING'] = dcorr, 'Width of correlator kernel [Hz].'
    hdr['BMAJ'] = bmaj
    hdr['BMIN'] = bmin
    hdr['BPA'] = bpa, 'Major axis, east from north [deg].'
    hdu[0].data = ccube

    # Save the file with the new filename.
    if output is None:
        output = fn.replace('.f', '_%.2f_%.2f_%.2f.f' % (bmin, bmaj, bpa))
    print("Saving convolved cube to %s%s." % (dir, output))
    hdu.writeto(dir+output, overwrite=True, output_verify='fix')
    return


def hanningkernel(fn, dcorr=2., npts=501, unit='chan'):
    """Returns the Hanning kernel with given width."""
    velax = readvelocityaxis(fn)
    dchan = abs(np.mean(np.diff(velax)))
    if unit == 'kHz':
        dcorr *= sc.c / fits.getval(fn, 'restfreq') / 1e3
    elif unit == 'chan':
        dcorr *= dchan
    elif unit != 'km/s':
        raise ValueError("Units must be 'kHz', 'chan' or 'km/s'.")
    hanning = interp1d(np.linspace(-2, 2, npts), np.hanning(npts),
                       bounds_error=False, fill_value=0.0)
    kern = [hanning(i * dchan / dcorr)
            for i in range(-velax.size, velax.size+1)
            if hanning(i * dchan / dcorr) > 0.0]
    return Kernel(np.squeeze(kern))


def readvelocityaxis(fn):
    """Wrapper for _velocityaxis and _spectralaxis."""
    if fits.getval(fn, 'ctype3').lower() == 'freq':
        specax = _spectralaxis(fn)
        try:
            nu = fits.getval(fn, 'restfreq')
        except KeyError:
            nu = fits.getval(fn, 'restfrq')
        return (nu - specax) * sc.c / nu / 1e3
    else:
        return _velocityaxis(fn)


def _velocityaxis(fn):
    """Return velocity axis in [km/s]."""
    a_len = fits.getval(fn, 'naxis3')
    a_del = fits.getval(fn, 'cdelt3')
    a_pix = fits.getval(fn, 'crpix3')
    a_ref = fits.getval(fn, 'crval3')
    return (a_ref + (np.arange(a_len) - a_pix + 1) * a_del) / 1e3


def _spectralaxis(fn):
    """Returns the spectral axis in [Hz]."""
    a_len = fits.getval(fn, 'naxis3')
    a_del = fits.getval(fn, 'cdelt3')
    a_pix = fits.getval(fn, 'crpix3')
    a_ref = fits.getval(fn, 'crval3')
    return a_ref + (np.arange(a_len) - a_pix + 1) * a_del


def readpositionaxis(fn):
    """Returns the position axis in ["]."""
    a_len = fits.getval(fn, 'naxis2')
    a_del = fits.getval(fn, 'cdelt2')
    a_pix = fits.getval(fn, 'crpix2')
    return 3600. * ((np.arange(1, a_len+1) - a_pix) * a_del)


def beamkernel(bmaj, bmin, bpa, dpix):
    """Returns the 2D Gaussian kernel."""
    fwhm2std = 2. * np.sqrt(2. * np.log(2))
    bmaj /= fwhm2std * dpix
    bmin /= fwhm2std * dpix
    return Kernel(gaussian2D(bmin, bmaj, pa=np.radians(bpa)))


def gaussian2D(dx, dy, pa=0.0, nsig=4):
    """2D Gaussian kernel in pixel coordinates."""
    xm = np.arange(-nsig*max(dy, dx), nsig*max(dy, dx)+1)
    x, y = np.meshgrid(xm, xm)
    x, y = rotate(x, y, pa)
    k = np.power(x / dx, 2) + np.power(y / dy, 2)
    return np.exp(-0.5 * k) / 2. / np.pi / dx / dy


def rotate(x, y, t):
    '''Rotation by angle t [rad].'''
    x_rot = x * np.cos(t) + y * np.sin(t)
    y_rot = y * np.cos(t) - x * np.sin(t)
    return x_rot, y_rot
