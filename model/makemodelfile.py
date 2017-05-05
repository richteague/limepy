"""
Part of the limepy package.

Functions to write the model.c file used for LIME. Note that in the case of
ortho-H2 and para-H2, they are density[0] and density[1] respectively.

For the optional values in limeclass.py there are two possible options
depending on the type: a single value denotes a disk-wide value, an array is
one that require interpolation.
"""

import os


def makeFile(m, model):
    """Write a model_m.c file."""
    tempfile = ['#include "lime.h"\n', '#include "math.h"\n',
                '#include "stdio.h"\n', '#include "stdlib.h"\n',
                '#include "%s"\n\n' % model.header.fn]
    writeModelProperties(tempfile, model)
    writeWeighting(tempfile, model)
    writeImageParameters(tempfile, m, model)
    writeFindValue(tempfile, model)
    writeDensity(tempfile, model)
    writeTemperatures(tempfile, model)
    writeAbundance(tempfile, model)
    writeGastoDust(tempfile, model)
    writeDoppler(tempfile, model)
    writeVelocityStructure(tempfile, model)
    with open('model_%d.c' % m, 'w') as tosave:
        for line in tempfile:
            tosave.write('%s' % line)
    return


def writeModelProperties(temp, model):
    """Radiative transfer properties."""
    temp.append('void input(inputPars *par, image *img){\n\n')
    temp.append('\tpar->radius = %.5f*AU;\n' % model.radius)
    temp.append('\tpar->minScale = %.5f*AU;\n' % model.minScale)
    temp.append('\tpar->pIntensity = %.d;\n' % model.pIntensity)
    temp.append('\tpar->sinkPoints = %.d;\n' % model.sinkPoints)
    temp.append('\tpar->sampling = %d;\n' % model.sampling)
    temp.append('\tpar->moldatfile[0] = "%s";\n' % model.moldatfile)
    temp.append('\tpar->dust = "%s";\n' % model.dust)
    temp.append('\tpar->lte_only = %d;\n' % model.lte_only)
    temp.append('\tpar->blend = %d;\n' % model.blend)
    temp.append('\tpar->antialias = %d;\n' % model.antialias)
    temp.append('\tpar->traceRayAlgorithm = %d;\n' % model.traceRayAlgorithm)
    temp.append('\tpar->nThreads = %d;\n' % model.nThreads)
    if model.gridOutFile:
        temp.append('\tpar->gridOutFiles[3] = "%s.ds";\n' % model.name)
    temp.append('\n')
    return


def writeWeighting(temp, model):
    """Abundance and dust weighting values."""
    for i, cId in enumerate(model.collpartIds):
        temp.append('\tpar->collPartIds[{}] = {};\n'.format(i, cId))
    for i, weight in enumerate(model.dustWeights):
        temp.append('\tpar->dustWeights[{}] = {};\n'.format(i, weight))
    for i, weight in enumerate(model.nMolWeights):
        temp.append('\tpar->nMolWeights[{}] = {};\n'.format(i, weight))
    return


def writeImageParameters(temp, m, model):
    """Image parameters."""
    for i, inc in enumerate(model.incl):
        for p, pa in enumerate(model.posang):
            for a, azi in enumerate(model.azimuth):
                for t, trans in enumerate(model.transitions):
                    nimg = t + (a + p + i) * model.ntra
                    nimg += (p + i) * model.nazi + i * model.npos
                    writeImageBlock(temp, nimg, m, inc, pa, azi, trans, model)
    temp.append('}\n\n\n')
    return


def writeImageBlock(temp, nimg, m, inc, pa, azi, trans, model):
    """Write an image block."""
    imgs = '\timg[%d].' % nimg
    filename = '%s_%.2f_%.2f_%.2f_%d' % (m, inc, pa, azi, trans)
    temp.append(imgs+'pxls = %d;\n' % model.pxls)
    temp.append(imgs+'imgres = %.3f;\n' % model.imgres)
    temp.append(imgs+'distance = %.3f*PC;\n' % model.distance)
    temp.append(imgs+'unit = %d;\n' % model.unit)
    temp.append(imgs+'filename = "%s.fits";\n' % filename)
    temp.append(imgs+'source_vel = %.3f;\n' % model.source_vel)
    temp.append(imgs+'nchan = %d;\n' % (model.nchan * model.oversample))
    temp.append(imgs+'velres = %.3f;\n' % (model.velres / model.oversample))
    temp.append(imgs+'trans = %d;\n' % trans)
    temp.append(imgs+'incl = %.3f;\n' % inc)
    temp.append(imgs+'posang = %.3f;\n' % (pa-1.57))
    temp.append(imgs+'azimuth = %.3f;\n\n' % azi)
    return


def writeCoords(temp, model):
    """Define the model coordinates for each function."""
    if not (model.coordsys is 'cylindrical' and model.ndim is 2):
        raise NotImplementedError
    if model.coordsys is 'cylindrical':
        if model.ndim is 2:
            temp.append('\tdouble c1 = sqrt(x*x + y*y) / AU;\n')
            temp.append('\tdouble c2 = fabs(z) / AU;\n')
            temp.append('\tdouble c3 = -1.;\n')
    return


def writeFindValue(temp, model):
    """Include the interpolation functions for array inputs."""
    if not (model.coordsys is 'cylindrical' and model.ndim is 2):
        raise NotImplementedError
    path = os.path.dirname(__file__)+'../interpolation/'
    with open(path+'%dD_%s.c' % (model.ndim, model.coordsys)) as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace('NCELLS', '%d' % model.ncells)
        temp.append(line)
    temp.append('\n\n')
    return


def writeDensity(temp, model):
    """Main collider densities."""
    temp.append('void density(double x, double y,')
    temp.append('double z, double *density) {\n\n')
    writeCoords(temp, model)
    for i, val in enumerate(model.rescaledens):
        temp.append('\tdensity[%d] = %.2f *' % (i, val))
        temp.append(' findvalue(c1, c2, c3, dens);\n')
        temp.append('\tif (density[%d] < 1e3)' % i)
        temp.append(' {\n\t\tdensity[%d] = 1e3;\n\t}\n\n' % i)
    temp.append('}\n\n\n')
    return


def writeTemperatures(temp, model):
    """Gas and dust temperatures."""
    temp.append('void temperature(double x, double y, double z,')
    temp.append('double *temperature) {\n\n')
    writeCoords(temp, model)
    temp.append('\ttemperature[0] = findvalue(c1, c2, c3, temp);\n')
    if model.rescaletemp:
        temp.append('\ttemperature[0] *= %.3f;\n' % model.rescaletemp)
    temp.append('\tif (temperature[0] < 0.0) ')
    temp.append('{\n\t\ttemperature[0] = 0.0;\n\t}\n\n')
    if type(model.dtemp) is float:
        temp.append('\ttemperature[1] = %.3f * temperature[0];' % model.dtemp)
        temp.append('\n\n')
    else:
        temp.append('\ttemperature[1] = findvalue(c1, c2, c3, dtemp);\n')
        temp.append('\tif (temperature[1] < 0.0) ')
        temp.append('{\n\t\ttemperature[1] = 0.0;\n\t}\n')
    temp.append('}\n\n\n')
    return


def writeAbundance(temp, model):
    """Molecular abundances."""
    temp.append('void abundance(double x, double y, double z,')
    temp.append(' double *abundance) {\n\n')
    if type(model.abund) is float:
        temp.append('\tabundance[0] = %.3e;\n' * model.abund)
    else:
        writeCoords(temp, model)
        temp.append('\tfindvalue(c1, c2, c3, abund);\n' % model.abund)
    if model.depletion:
        temp.append('\tabundance[0] *= %.3e;\n' % model.depletion)
    temp.append('\tif (abundance[0] < 0.){\n\t\tabundance[0] = 0.;\n\t}\n')
    temp.append('\n}\n\n\n')
    return


def writeDoppler(temp, model):
    """Doppler broadening component."""
    temp.append('void doppler(double x, double y,')
    temp.append('double z, double *doppler) {\n\n')
    if type(model.turb) is float:
        temp.append('\t*doppler = %.3e;\n' % model.turb)
    else:
        writeCoords(temp, model)
        temp.append('\t*doppler = findvalue(c1, c2, c3, turb);\n')
    if model.turbtype == 'mach':
        temp.append('\tdouble val[2];\n')
        temp.append('\ttemperature(x, y, z, &val);\n')
        temp.append('\t*doppler *= sqrt(KBOLTZ * val[0] / 2.34 / AMU);\n')
    temp.append('\n}\n\n\n')
    return


def writeGastoDust(temp, model, ming2d=1.):
    """Gas-to-dust ratios."""
    temp.append('void gasIIdust(double x, double y,')
    temp.append(' double z, double *gtd) {\n\n')
    if type(model.g2d) is float:
        temp.append('\t*gtd = %.1f;\n\n' % model.g2d)
    else:
        writeCoords(temp, model)
        temp.append('\t*gtd = findvalue(c1, c2, c3, g2d);\n\n')
    temp.append('\tif (*gtd < 1e-4) {\n\t\t*gtd = 1e-4;\n\t}\n}\n\n\n')
    return


def writeVelocityStructure(temp, model):
    """Velocity component."""
    temp.append('void velocity(double x, double y,')
    temp.append('double z, double *velocity) {\n\n')
    temp.append('\tif (sqrt(x*x + y*y + z*z) == 0.0){\n')
    temp.append('\t\tvelocity[0] = 0.0;\n')
    temp.append('\t\tvelocity[1] = 0.0;\n')
    temp.append('\t\tvelocity[2] = 0.0;\n')
    temp.append('\t\t return;\n')
    temp.append('\t}\n\n')
    temp.append('\tvelocity[0] = sqrt(6.67e-11 * ')
    temp.append('%.3f * 1.989e30 / ' % model.mstar)
    temp.append('sqrt(x*x + y*y + z*z));\n')
    temp.append('\tvelocity[0] *= sin(atan2(y,x));\n')
    temp.append('\tvelocity[1] = sqrt(6.67e-11 * ')
    temp.append('%.3f * 1.989e30 / ' % model.mstar)
    temp.append('sqrt(x*x + y*y + z*z));\n')
    temp.append('\tvelocity[1] *= cos(atan2(y,x));\n')
    temp.append('\tvelocity[2] = 0.0;\n')
    temp.append('\n}\n\n\n')
    return