"""
Part of the limepy package.

Tools to read in the collisional rate data from LAMDA.

Functions to do:

    > Tidy up collisional rates.
    > Consider non-basic linear rotators.

"""

import os
import numpy as np


class level:
    """Energy levels of the molecule."""
    def __init__(self, E, g, J):
        self.E = float(E)
        self.g = int(g)
        self.J = int(J)
        return


class lines:
    """Radiative transitions of the molecule."""
    def __init__(self, i, j, A, freq, Eup):
        self.i = int(i)  # Upper energy level.
        self.j = int(j)  # Lower energy level.
        self.A = float(A)
        self.freq = float(freq) * 1e9
        self.Eup = float(Eup)
        return


class crates:
    """Collisional rates of the molecule and given partner."""
    def __init__(self, trans, temps, tup, tlo, rates):
        self.trans = trans
        self.temps = temps
        self.tup = tup
        self.tlo = tlo
        self.rates = rates
        return


class ratefile:
    """LAMDA collisional rate data file."""
    def __init__(self, molecule, verbose=False):
        self.verbose = verbose

        # Initially search for the molecule in the aux directory.
        # If not there, assume it is a direct path.

        self.path = os.path.dirname(__file__)
        self.aux = self.path.replace('analysis', 'aux/')
        rates = '%s.dat' % molecule.lower()
        if rates in os.listdir(self.aux):
            fn = self.aux+rates
        else:
            fn = molecule
        with open(fn) as f:
            self.filein = f.readlines()

        # Basic data about the molecule.

        self.molecule = self.filein[1].strip()
        self.mu = float(self.filein[3].strip())
        self.nlev = int(self.filein[5].strip())
        if self.verbose:
            print('Molecule: %s.' % self.molecule)
            print('Molecular weight: %d.' % self.mu)
            print('Energy levels: %d.' % self.nlev)

        # Energy levels.

        self.levels = {}
        for line in range(self.nlev):
            self.populate_levels(self.filein[7+line].strip())

        # Radiative transitions.

        self.nlin = int(self.filein[8+self.nlev])
        if self.verbose:
            print('Radiative transitions: %d.' % self.nlin)
        self.lines = {}
        for line in range(self.nlin):
            self.populate_lines(self.filein[10+self.nlev+line])

        # Collisional rates. Will loop through all the found collisional rates.

        self.ncoll = int(self.filein[11+self.nlev+self.nlin])
        if self.verbose:
            print('Number of collision partners: %d.' % self.ncoll)
        self.rates = {}
        self.cl = 13 + self.nlev + self.nlin

        for i in range(self.ncoll):
            ID = int(self.filein[self.cl][0])
            name = coll_ID[ID]
            ntrans = int(self.filein[self.cl+2])
            temps = self.filein[self.cl+6]
            temps = [float(x) for x in temps.split(' ') if x != '']
            temps = np.array(temps)
            self.populate_collisions(self.cl, ID, ntrans, temps)
            if self.verbose:
                s = 'Collisions with %s ' % name
                s += 'have %d transitions ' % ntrans
                s += 'at %d temperatures.' % temps.size
                print(s)
            self.cl += ntrans + 9

        self.partners = [x for x in self.rates.keys() if type(x) == str]

        return

    def populate_levels(self, line):
        """Populate self.lines."""
        L, E, g, J = [float(x) for x in line.split(' ') if x != '']
        self.levels[int(L)] = level(E, g, J)
        return

    def populate_lines(self, line):
        """Populate self.transitions."""
        line = [float(x) for x in line.split(' ') if x != '']
        transition, i, j, A, freq, Eup = line
        self.lines[int(transition)] = lines(i, j, A, freq, Eup)
        return

    def populate_collisions(self, cl, ID, ntrans, temps):
        """Populate self.collisions."""
        lst = self.filein[self.cl+8:self.cl+8+ntrans]
        lst = np.array([np.fromstring(lst[0], sep=' ') for l in lst]).T
        trans = lst[0]
        tup = lst[1]
        tlo = lst[2]
        rates = lst[3:]
        self.rates[ID] = crates(trans, temps, tup, tlo, rates)
        self.rates[coll_ID[ID]] = self.rates[ID]
        return


coll_ID = {}
names = ['H2', 'pH2', 'oH2', 'e', 'H', 'He', 'Hplus']
for i, name in enumerate(names):
    coll_ID[name] = i + 1
    coll_ID[i + 1] = name
