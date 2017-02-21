"""
Part of the limepy package.

Functions and classes to ease reading and parsing the collisional rates used
for LIME obtained from LAMDA (http://home.strw.leidenuniv.nl/~moldata/).

Functions to do:

    > Tidy everything up.

From LAMDA, the format of the files:

Lines 1-2: molecule (or atom) name
Lines 3-4: molecular (or atomic) weight (a.m.u.)
Lines 5-6: number of energy levels (NLEV)
Lines 7-7+NLEV: level number, level energy (cm-1), statistical weight. These
numbers may be followed by additional info such as the quantum numbers, which
are however not used by the program. The levels must be listed in order of
increasing energy.
Lines 8+NLEV-9+NLEV: number of radiative transitions (NLIN)
Lines 10+NLEV-10+NLEV+NLIN: transition number, upper level, lower level,
spontaneous decay rate (s-1). These numbers may be followed by additional info
such as the line frequency, which is however not used by the program.
Lines 11+NLEV+NLIN-12+NLEV+NLIN: number of collision partners
Lines 13+NLEV+NLIN-14+NLEV+NLIN: collision partner ID and reference. Valid
identifications are: 1=H2, 2=para-H2, 3=ortho-H2, 4=electrons, 5=H, 6=He.
Lines 15+NLEV+NLIN-16+NLEV+NLIN: number of transitions for which collisional
data exist (NCOL)
Lines 17+NLEV+NLIN-18+NLEV+NLIN: number of temperatures for which collisional
data exist
Lines 19+NLEV+NLIN-20+NLEV+NLIN: values of temperatures for which collisional
data exist
Lines 21+NLEV+NLIN-21+NLEV+NLIN+NCOL: transition number, upper level, lower
level; rate coefficients (cm3s-1) at each temperature.

"""

import numpy as np


class ratefile:

    names = ['H2', 'pH2', 'oH2', 'electrons', 'H', 'He', 'H+']
    ID = {name: i for name, i in enumerate(names)}

    def __init__(self, fn):
        with open(fn) as f:
            self.filein = f.readlines()


        self.molecule = self.filein[1].strip()
        self.mu = float(self.filein[3].strip())
        self.nlev = int(self.filein[5].strip())

        self.nlevels = int(self.filein[5].strip())
        self.ntransitions = int(self.filein[8+self.nlevels].strip())
        self.npartners = int(self.filein[11+self.nlevels+self.ntransitions].strip())
        sself.trans, self.up, self.down, self.A, self.freq, self.Eup = readin
        # Read in the partner names and the bounding line values.
        self.partners = []
        self.linestarts = []
        self.lineends = []
        n = 0
        linestart = 12+self.nlevels+self.ntransitions
        while n < self.npartners:
            self.linestarts.append(linestart)
            names = np.array(['H2', 'pH2', 'oH2', 'electrons', 'H', 'He', 'H+'])
            name = names[int(self.filein[linestart+1][0])-1]
            lineend = linestart+9+int(self.filein[linestart+3].strip())
            self.partners.append(name)
            self.lineends.append(lineend)
            linestart = lineend
            n += 1

        # Read in the energy level structure.
        self.levels = self.filein[7:7+self.nlevels]
        self.levels = np.array([[float(n) for n in levelsrow.strip().split()]
                                 for levelsrow in self.levels]).T

        # Read in the radiative transitions.
        self.transitions = self.filein[10+self.nlevels:10+self.nlevels+self.ntransitions]
        self.transitions = np.array([[float(n) for n in transrow.strip().split()]
                                      for transrow in self.transitions]).T

        # Split into appropriate arrays.
        self.deltaE = self.levels[1]
        self.weights = self.levels[2]
        self.J = self.levels[3]
        self.EinsteinA = self.transitions[3]
        self.frequencies = self.transitions[4] * 1e9
        self.E_upper = self.transitions[5]

        return
