"""
Part of the limepy package.

Tools to read in the collisional rate data from LAMDA.

Functions to do:

    > Tidy up collisional rates.
    > Consider non-basic linear rotators.

"""

import os
import numpy as np

# -- Collisional Partners --
#
# Diciontary to swap between collisional partner names and LAMDA values.

coll_ID = {}
names = ['H2', 'pH2', 'oH2', 'e', 'H', 'He', 'Hplus']
for i, name in enumerate(names):
    coll_ID[name] = i + 1
    coll_ID[i + 1] = name


class energylevel:
    """Energy levels of the molecule."""
    def __init__(self, l, E, g, J):
        self.level = int(l)
        self.E = float(E)
        self.g = int(g)
        self.J = int(J)
        return


class transition:
    """Radiative transitions of the molecule."""
    def __init__(self, t, Jup, Jlo, A, freq, Eup):
        self.trans = int(t)
        self.Jup = int(Jup)
        self.Jlo = int(Jlo)
        self.A = float(A)
        self.freq = float(freq) * 1e9
        self.Eup = float(Eup)
        return


class collrates:
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

        self.aux = '/Users/richardteague/PythonPackages/limepy/aux/'
        rates = '%s.dat' % molecule.lower()
        if rates in os.listdir(self.aux):
            fn = self.aux+rates
        else:
            fn = molecule

        with open(fn) as f:
            self.filein = f.readlines()

        self.molecule = self.filein[1].strip()
        self.mu = float(self.filein[3].strip())
        self.nlev = int(self.filein[5].strip())
        if self.verbose:
            print('Molecule: %s.' % self.molecule)
            print('Molecular weight: %d.' % self.mu)
            print('Energy levels: %d.' % self.nlev)

        self.levels = {}
        for line in range(self.nlev):
            self.populate_levels(self.filein[7+line].strip())

        self.nlin = int(self.filein[8+self.nlev])
        if self.verbose:
            print('Radiative transitions: %d.' % self.nlin)
        self.transitions = {}
        for line in range(self.nlin):
            self.populate_transitions(self.filein[10+self.nlev+line])

        self.ncoll = int(self.filein[11+self.nlev+self.nlin])
        if self.verbose:
            print('Number of collision partners: %d.' % self.ncoll)
        self.collisions = {}
        self.cl = 13 + self.nlev + self.nlin

        for i in range(self.ncoll):
            # Populate the collisional rates.
            ID = int(self.filein[self.cl][0])
            name = coll_ID[ID]
            ntrans = int(self.filein[self.cl+2])
            temps = self.filein[self.cl+6]
            temps = [float(x) for x in temps.split(' ') if x != '']
            temps = np.array(temps)
            if self.verbose:
                s = 'Collisions with %s ' % name
                s += 'have %d transitions ' % ntrans
                s += 'at %d temperatures.' % temps.size
                print(s)
            self.populate_collisions(self.cl, ID, ntrans, temps)
            self.cl += ntrans + 9

        return

    def populate_levels(self, line):
        """Populate self.lines."""
        L, E, g, J = [float(x) for x in line.split(' ') if x != '']
        self.levels[int(L)] = energylevel(L, E, g, J)
        return

    def populate_transitions(self, line):
        """Populate self.transitions."""
        line = [float(x) for x in line.split(' ') if x != '']
        t, Jup, Jlo, A, freq, Eup = line
        self.transitions[int(t)] = transition(t, Jup, Jlo, A, freq, Eup)
        return

    def populate_collisions(self, cl, ID, ntrans, temps):
        """Populate self.collisions."""
        rates = self.filein[self.cl+8:self.cl+8+ntrans]
        rates = np.array([np.fromstring(rates[0], sep=' ') for l in rates]).T
        trans = rates[0]
        tup = rates[1]
        tlo = rates[2]
        rates = rates[3:]
        self.collisions[ID] = collrates(trans, temps, tup, tlo, rates)
        self.collisions[coll_ID[ID]] = self.collisions[ID]
        return
