# Functions to make a header file suitable for makeLIME.py
# Call from the command line:
# >>> python makeheader.py path/to/file name(optional)

import sys
import numpy as np
import scipy.constants as sc


def writeHeaderString(array, name):

    # Write the array strings.

    tosave = 'const static double %s[%d] = {' % (name, array.size)
    for val in array:
        tosave += '%.3e, ' % val
    tosave = tosave[:-2] + '};\n'
    return tosave


def makeHeader(path, name=None):

    # Write the header.
    # Make sure the orientation is correct.
    # Remove values with a zero density.

    if not type(path) is str:
        raise TypeError("Must be path to chemical model.")
    if path.endswith('.out'):
        data = np.loadtxt(path, skiprows=3).T
    else:
        data = np.loadtxt(path).T
    data = np.array([param[data[2] > 0] for param in data])
    if data.shape[0] > data.shape[1]:
        data = data.T

    # Check that the arrays are correctly sized.
    # If it is the 8 parameter value, remove the average grainsize.

    if data.shape[0] == 8:
        data = np.vstack([data[:5], data[6:]])
        datalong = True
    elif data.shape[0] == 5:
        datalong = False
    else:
        raise ValueError("Must be either a 5 or 8 column file.")

    # Make the conversions to LIME appropriate units:
    # Main collider density (H2) is in [m^-3].
    # Relative abundance is with respect to the main collider density.
    # Temperatures are all in [K].

    with np.errstate(divide='ignore'):
        data[2] /= 2.37 * sc.m_p * 1e3
        data[-1] = data[-1]/data[2]
        data[2] *= 1e6
        if datalong:
            data[5] = np.where(data[5] != 0.0, 1./data[5], 100.)
        data = np.where(~np.isfinite(data), 0.0, data)

    # Make sure that each column has a cell at the midplane: z = 0.

    for r in np.unique(data[0]):
        idx = data[1][data[0] == r].argmin()
        data[1][idx] = 0.0

    # Make sure these are the same as in the template file.
    # Note that we skip the average grainsize array.

    if datalong:
        arrnames = ['c1arr', 'c2arr', 'dens', 'temp', 'dtemp',
                    'gastodust', 'abund']
    else:
        arrnames = ['c1arr', 'c2arr', 'dens', 'temp', 'abund']

    # Write the arrays and output to file.

    hstring = ''
    for i, array in enumerate(data):
            hstring += writeHeaderString(array, arrnames[i])

    if name is None:
        name = path.split('/')[-1]
        i = -1
        while name[i] != '.':
            i -= 1
            if i == 0:
                i = len(name)
                break
        name = name[:i]
    elif name[-2:] == '.h':
        name = name[:-2]

    with open('%s.h' % name, 'w') as hfile:
        hfile.write('%s' % hstring)
    print "Written to '%s.h'." % name

    return

if len(sys.argv) == 2:
    makeHeader(sys.argv[1])
else:
    makeHeader(sys.argv[1], sys.argv[2])
