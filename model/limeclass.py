"""
Part of the limepy package.

A class defining the models to run with LIME. Should first check that all the
appropriate files are available, i.e. the dust opacities, collisional rates and
the header file. If they are, create a new folder and copy all of that into
there.

The header file should provide the default values for the functions required by
LIME. They will be read from the array names and must be one of:

    'c1arr' - coordinate
    'c2arr' - coordinate
    'dens'  - main collider density [/m^3]
    'temp'  - gas temperature [K]
    'dtemp' - dust temperature [K] (optional)
    'abund' - relative abundance of radiating species. (optional)
    'g2d'   - gas to dust ratio (optional)
    'turb'  - non-thermal component [m/s or Mach] (optional)

In addition, the values of 'c1arr' and 'c2arr' will be used to estimate the
values for 'par->radius' and 'par->minScale'.

If no dust temperature is given then it is assumed that gas and dust
temperatures are equal. If no relative abudance is given, a homogeneous 1e-5 is
assumed If no gas-to-dust ratio is given, it is assumed to be 100. If no
turbulence is given then it is assumed to be a laminar disk. Note that the
units of the turbulence should be given as an extra parameter, otherwise it is
assumed to be [m/s].

From the rates file will check whether H2 or oH2 and pH2 are available. If both
ortho-H2 and para-H2 collisional rates these will be used in preference over H2
rates. The density will be split using the 'opr' value. The user can try and
force the use of H2 with the 'useH2' keyword.

Functions to do:

    > Allow the inclusion of 'vel1', 'vel2' and 'vel3' in the header.
    > Allow for other coordinate systems other than cylindrical.
    > Allow for non-H2 main colliders.
    > Take advantage of multiple units to be ray traced.
    > Include a temperature dependent ortho to para ratio.

"""

import os
from headerclass import readheader
from limepy.analysis.collisionalrates import readrates
import numpy as np


class model:

    def __init__(self, header, rates, dust='jena_thin_e6.tab', **kwargs):

        self.path = os.path.dirname(__file__)
        self.aux = self.path + '../aux/'
        self.directory = kwargs.get('directory', '../')

        # Check whether the required files are there. If so, move them.

        if not os.path.isfile('../'+header):
            raise ValueError('No header file found.')
        if not os.path.isfile(self.aux+rates):
            raise ValueError('No collisional rates found.')
        if not os.path.isfile(self.aux+dust):
            raise ValueError('No dust opacities found.')
        os.system('cp ../%s .' % header)
        os.system('cp %s%s .' % (self.aux, rates))
        os.system('cp %s%s .' % (self.aux, dust))
        self.header = readheader(header)
        self.rates = readrates(rates)
        self.moldatfile = rates
        self.dust = dust

        # Extract information from the header file.

        self.c1arr = self.header.params['c1arr']
        self.c2arr = self.header.params['c2arr']
        self.dens = self.header.params['dens']
        self.temp = self.header.params['temp']
        try:
            self.abund = self.header.params['abund']
        except:
            self.abund = kwargs.get('abund', 1e-4)
        try:
            self.g2d = self.header.params['g2d']
        except:
            self.g2d = kwargs.get('g2d', 100.)
        try:
            self.turb = self.header.params['turb']
        except:
            self.turb = kwargs.get('turb', 0.0)
        self.turbtype = kwargs.get('turbtype', 'absolute')
        if self.turbtype not in ['absolute', 'mach']:
            raise ValueError()
        try:
            self.dtemp = self.header.params['dtemp']
        except:
            self.dtemp = kwargs.get('dtemp', 1.0)
        self.mstar = float(kwargs.get('mstar', 0.7))

        self.radius = self.header.rmax
        self.minScale = max(self.header.rmin, 1e-4)
        if self.minScale >= self.radius:
            raise ValueError('radius < minScale')

        self.ndim = self.header.ndim
        self.ncells = self.header.ncells
        self.coordsys = self.header.coordsys

        # Collisional rates. Can try to force the use of H2 over oH2 and pH2.
        # The ortho / para ratio must be iterable for the make model file and
        # is changed to the rescaling factor, e.g. for an ortho/para ratio of
        # 3, opr = [0.75, 0.25].

        self.H2 = 'H2' in self.rates.partners
        self.oH2 = 'oH2' in self.rates.partners
        self.pH2 = 'pH2' in self.rates.partners
        self.opr = kwargs.get('opr', 3.)

        if kwargs.get('onlyH2', False) and self.H2:
            self.oH2 = False
            self.pH2 = False

        if (self.oH2 and self.pH2):
            print 'Using oH2 and pH2 as collision partners.'
            self.rescaledens = np.array([self.opr, 1.]) / (1. + self.opr)
            self.collpartIds = [3, 2]
        elif self.H2:
            print 'Using H2 as single collision partner.'
            self.rescaledens = [1.0]
            self.collpartIds = [1]
        else:
            raise ValueError()

        self.nMolWeights = np.array([1.0 for cId in self.collpartIds])
        self.dustWeights = np.array([1.0 for cId in self.collpartIds])

        # Model parameters. These should affect the running of the model.

        self.pIntensity = float(kwargs.get('pIntensity', 1e5))
        self.sinkPoints = float(kwargs.get('sinkPoints', 3e3))
        if self.sinkPoints > self.pIntensity:
            print("Warning: sinkPoints > pIntensity.")

        self.samplingAlgorithm = int(kwargs.get('samplingAlgorithm', 0))
        if self.samplingAlgorithm not in [0, 1]:
            raise ValueError('samplingAlgorithm must be 0 or 1.')
        self.sampling = int(kwargs.get('sampling', 2))
        if self.sampling not in [0, 1, 2]:
            raise ValueError('sampling must be 0, 1 or 2.')

        self.gridOutFile = bool(kwargs.get('gridOutFile', False))

        self.lte_only = int(kwargs.get('lte_only', 1))
        if self.lte_only > 1:
            raise ValueError('lte_only must be 0 or 1.')
        elif self.lte_only == 0:
            print 'Running non-LTE model. Will be slow.'

        self.blend = int(kwargs.get('blend', 0))
        if self.blend > 1:
            raise ValueError('blend must be 0 or 1.')
        elif self.blend == 0:
            print 'Including line blending. Will be slow.'

        self.traceRayAlgorithm = int(kwargs.get('traceRayAlgorithm', 0))
        if self.traceRayAlgorithm > 1:
            raise ValueError('traceRayAlgorithm must be 0 or 1.')

        self.antialias = int(kwargs.get('antialias', 1))
        self.nSolveIters = kwargs.get('nSolveIters', None)
        self.nThreads = int(kwargs.get('nThreads', 20))

        # Image parameters. Inclinations, position angles, azimuthal angle and
        # transitions can be iterable. All permutations will be run.

        self.nchan = int(kwargs.get('nchan', 100))
        self.velres = float(kwargs.get('velres', 200.))
        self.pxls = int(kwargs.get('pxls', 128))
        self.distance = float(kwargs.get('distance', 1.))
        self.source_vel = kwargs.get('source_vel', 0.0)
        self.imgres = float(kwargs.get('imgres', None))
        if self.imgres is None:
            self.imgres = 2. * self.radius / self.distance / self.pxls
        self.unit = int(kwargs.get('unit', 1))
        if self.unit not in [0, 1, 2, 3]:
            raise ValueError('unit must be 0, 1, 2 or 3.')

        self.name = kwargs.get('name', header)
        self.name = self.name.split('/')[-1]
        if '.' in self.name:
            self.name = ''.join(self.name.split('.')[:-1])

        self.transitions = kwargs.get('transitions', [0.])
        self.transitions = checkiterable(self.transitions)
        self.ntra = len(self.transitions)

        self.incl = kwargs.get('incl', [0.])
        self.incl = checkiterable(self.incl)
        self.ninc = len(self.incl)

        self.posang = kwargs.get('posang', [0.])
        self.posang = checkiterable(self.posang)
        self.npos = len(self.posang)

        self.azimuth = kwargs.get('azimuth', [0.])
        self.azimuth = checkiterable(self.azimuth)
        self.nazi = len(self.aximuth)

        # Additional variables.

        self.cleanup = kwargs.get('cleanup', True)
        self.nmodels = int(kwargs.get('nmodels', 1))
        self.returnnoise = kwargs.get('returnnoise', False)
        if self.returnnoise and self.nmodels == 1:
            raise ValueError()
        self.rescaletemp = kwargs.get('rescaletemp', False)
        self.depletion = float(kwargs.get('depletion', False))
        self.oversample = int(kwargs.get('oversample', False))
        self.niceness = kwargs.get('niceness', False)
        self.waittime = kwargs.get('wait', 60.)

        # Additional variables to be updated.

        self.tcmb = kwargs.get('tcmb', None)
        if self.tcmb is not None:
            raise NotImplementedError()

        self.molI = kwargs.get('molI', None)
        if self.molI is not None:
            raise NotImplementedError()

        self.freq = kwargs.get('freq', None)
        if self.freq is not None:
            raise NotImplementedError()

        self.bandwidth = kwargs.get('bandwidth', None)
        if self.bandwidth is not None:
            raise NotImplementedError()

        self.gridInFile = kwargs.get('gridInFile', None)
        if self.gridInFile is not None:
            raise NotImplementedError()

        self.gridDensity = kwargs.get('gridDensity', None)
        if self.gridDensity is not None:
            raise NotImplementedError()
        return


def checkiterable(val):
    """Return val such that it can be iterated over."""
    try:
        iter(val)
    except:
        return [val]
    return val