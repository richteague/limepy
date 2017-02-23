"""
Part of the limepy package.

Class to read in and extract information from the C header files used to
include models into LIME.
"""

import numpy as np


class readheader:

    def __init__(self, filename, coordsys='cylindrical'):

        if coordsys == 'cylindrical':
            self.coordsys = coordsys
        else:
            raise NotImplementedError("Only 'cyclindrical'.")

        self.fn = filename.split('/')[-1]
        self.path = filename.replace(self.fn, '')
        if self.fn[-2:] != '.h':
            raise ValueError('headerfile must have a *.h extention.')

        # Read in the header and parse all the information.

        with open(self.path+self.fn) as f:
            self.hdr = f.readlines()

        nlines = len(self.hdr)
        parsed = [self.parse_line(line) for line in self.hdr]

        self.anames = np.array([parsed[i][0] for i in range(nlines)])
        self.ncells = np.mean([parsed[i][1] for i in range(nlines)])
        self.params = {name: np.array(parsed[i][2])
                       for i, name in enumerate(self.anames)}

        # Determine the number of dimensions.

        if 'c1arr' not in self.anames:
            raise ValueError('No c1arr found.')
        if 'c2arr' not in self.anames:
            raise ValueError('No c2arr found.')
        if 'c3arr' in self.anames:
            self.ndim = 3
        else:
            self.ndim = 2

        # Estimate the inner and outer radii required for LIME.

        self.rmin, self.rmax = self.estimate_grids()
        return

    def estimate_grids(self):
        """Return the minimum and maximum radial points [au]."""
        if self.coordsys == 'polar':
            return self.params['c1arr'].min(), self.params['c2arr'].max()
        rvals = np.hypot(self.params['c1arr'], self.params['c2arr'])
        return rvals.min(), rvals.max()

    def parse_line(self, line):
        """Parses a line."""

        # Splits 'name[ncells]'.
        namencells = line.split(' ')[3]
        for i, c in enumerate(namencells):
            if c == '[':
                name = namencells[:i]
                ii = i + 1
            if c == ']':
                ncells = int(namencells[ii:i])

        # Splits '{val, val, val, ..., val};'.
        vals = line.split('{')[-1]
        vals = vals.split('}')[0]
        vals = np.array([float(v) for v in vals.split(',')])
        return name, ncells, vals
