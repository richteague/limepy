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
    retcube [optional]: If

    -- Returns --

    ccube [optional]    If returncube, will return the convolved cube.

"""

import numpy as np
from astropy.io import fits
from astropy.convolution import Kernel, convolve, convolve_fft


def convolvecube(path, bmaj, bmin=None, bpa=0.0, fast=True, output=None):
    """Convolve a LIME model with a 2D Gaussian beam."""
    fn = path.split('/')[-1]
    dir = '/'.join(path.split('/')[:-1])+'/'
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

    # Calculate the beam kernel and convolve the cube.
    beam = beamkernel(bmaj, bmin, bpa, dpix)
    print("Beginning convolution...")
    if fast:
        ccube = np.array([convolve_fft(c, beam) for c in data])
    else:
        ccube = np.array([convolve(c, beam) for c in data])

    # Use the header of the input cube as the basis for the new cube.
    # Add in additional header keywords describing the beam.
    hdu = fits.open(path)
    hdr = hdu[0].header
    hdr['BMAJ'] = bmaj
    hdr['BMIN'] = bmin
    hdr['BPA'] = bpa
    hdu[0].data = ccube

    # Save the file with the new filename.
    if output is None:
        output = fn.replace('.f', '_%.2f_%.2f_%.2f.f' % (bmin, bmaj, bpa))
    print("Saving convolved cube to %s%s." % (dir, output))
    hdu.writeto(dir+output, overwrite=True, output_verify='fix')
    return


def readpositionaxis(fn):
    """Returns the position axis in ["]."""
    a_len = fits.getval(fn, 'naxis2')
    a_del = fits.getval(fn, 'cdelt2')
    a_pix = fits.getval(fn, 'crpix2')
    return 3600. * ((np.arange(1, a_len+1) - a_pix + 0.5) * a_del)


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
