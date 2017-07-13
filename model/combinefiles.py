"""
Part of the limepy package.

Functions to help combine the multiple runs and provide statistics of the
ensemble.
"""

import os
import numpy as np
from astropy.io import fits


def filename(prefix, i, p, a, t):
    """Returns the filename"""
    return '{}_{:.2f}_{:.2f}_{:.2f}_{:d}.fits'.format(prefix, i, p, a, t)


def moveFiles(model, prefix='0_', suffix='.fits'):
    """Move the finished models to appropriate end directory."""
    files = [fn for fn in os.listdir('./') if fn.endswith(suffix)]
    for fn in [fn for fn in files if fn.startswith(prefix)]:
        os.system('mv %s ../%s' % (fn, model.name+fn[1:]))
    return


def moveGrids(model):
    """Move the model grids."""
    if not model.gridOutFile:
        return
    files = [fn for fn in os.listdir('./') if fn.endswith('.ds')]
    for fn in files:
        os.system('mv %s ../' % fn)
    return


def averageModels(model):
    """Average over all the models and save to 0_*.fits."""
    for i in model.incl:
        for p in model.posang:
            for a in model.azimuth:
                for t in model.transitions:
                    fn = filename(0, i, p, a, t)
                    if model.nmodels > 1:
                        hdu = fits.open(fn)
                        avg = [fits.getdata(filename(m, i, p, a, t))
                               for m in range(model.nmodels)]
                        getNoise(avg, i, p, a, t, model)
                        hdu[0].data = np.average(avg, axis=0)
                        try:
                            hdu.writeto(fn, overwrite=True)
                        except TypeError:
                            hdu.writeto(fn, clobber=True)
                    writeFitsHeader(fn, model, i, p, a)
    return


def writeFitsHeader(filename, model, inc, pa, azi):
    """Include model data in the final .fits file header."""
    data, header = fits.getdata(filename, header=True)
    header['DISTANCE'] = model.distance, 'Distance in parsec.'
    header['MODEL'] = model.header.fn.split('/')[-1], 'Input model.'
    header['INC'] = inc, 'Inclianation in radians.'
    header['PA'] = pa, 'Position angle in radians.'
    header['AZI'] = azi, 'Azimuthal angle in radians.'
    header['NMODELS'] = model.nmodels, 'Number of models averaged.'
    header['OPR'] = model.opr, 'Ortho-para ratio of H2.'
    header['MSTAR'] = model.mstar, 'Mass of central star in Msun.'
    try:
        fits.writeto(filename, data, header, overwrite=True)
    except TypeError:
        fits.writeto(filename, data, header, clobber=True)
    return


def getNoise(avgmodels, i, p, a, t, model):
    """Save the standard deviation of each voxel."""
    if not model.returnnoise:
        return
    noise = np.nanstd(avgmodels, axis=0)
    hdu = fits.open(filename(0, i, p, a, t))
    hdu[0].data = noise
    hdu.writeto(filename(0, i, p, a, t).replace('.fits', '_noise.fits'))
    return
