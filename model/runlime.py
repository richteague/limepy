"""
Part of the limepy package.

Run (multiple) LIME models in a tidy way.
"""

import os
import time
import makemodelfile as make
import combinefiles as comb
import limeclass as lime


def run_model(headerfile, moldatfile, **kwargs):
    """ Run a LIME model. Check readme for possible kwargs and defaults."""

    # Start the clock to time the running of models.

    print '\n'
    t0 = time.time()
    folder_name = makeUniqueFolder()
    os.chdir(folder_name)

    # Generate a LIME model class instance.

    model = lime.model(headerfile, moldatfile, **kwargs)
    print '\nInput is in %dD-%s coordinates.' % (model.ndim, model.coordsys)
    print 'Assuming a value of minScale = %.2f au.' % model.minScale
    print 'Assuming a value of radius = %.2f au.' % model.radius

    # For each iteration, run a model with a pause of waittime seconds.

    print '\n'
    for m in range(model.nmodels):
        print 'Running model %d of %d.' % (m+1, model.nmodels)
        make.makeFile(m, model)
        cmd = 'screen -d -m lime -n model_%d.c' % m
        if model.niceness:
            cmd = 'nice -n %d ' % model.niceness + cmd
        os.system(cmd)
        time.sleep(model.waittime)

    # Make sure all the models have run.
    # Check the number of *.x files to guess how many are running.
    # If nohup_X.out contains segmentation fault, then quit.

    remaining = -1
    print '\n'
    while len([fn for fn in os.listdir('./') if fn.endswith('.x')]) > 0:
        nremaining = len([fn for fn in os.listdir('./') if fn.endswith('.x')])
        if nremaining != remaining:
            print 'Waiting on %d model(s) to run.' % nremaining
            remaining = nremaining
        time.sleep(10.*remaining)

        # Check for segmentation faults.
        # If the number of nohup_X.out files with 'core dumped' in them
        # equals the number of lime_$$.x files left, assumped all have quit.

        nohups = [fn for fn in os.listdir('./') if fn.startswith('nohup')]
        coresdumped = 0
        for nh in nohups:
            if 'core dumped' in open(nh).read():
                coresdumped += 1
        if coresdumped == nremaining:
            print 'Found %d segmentation faults.' % coresdumped
            break

    models_run = len([fn for fn in os.listdir('./') if fn.endswith('.fits')])
    if models_run < model.nmodels:
        print 'Not all models were successfully run.'
        print 'Aborting without clean-up.\n'
        return
    else:
        print 'All instances complete.\n'

    # Combine the model ensemble. Here we remove all the output grids. As they
    # all have the same name, they will just be overwritten.

    if model.oversample > 1 or model.hanning:
        comb.downsampleModels(model)
    comb.averageModels(model)
    comb.moveFiles(model, suffix='.fits')
    comb.moveGrids(model)

    # Clean up.
    os.chdir('../')
    if model.cleanup:
        print 'Cleaning up temporary folders.'
        os.system('rm -rf %s' % folder_name)

    # Print the total time.
    print 'Finished in %s.\n\n' % seconds2hms(time.time() - t0)

    return


def makeUniqueFolder(fname='tempfolder', path='./'):
    """Make a folder with a unique name."""
    if not os.path.isdir(path+fname):
        os.makedirs(path+fname)
    else:
        suffix = 1
        while os.path.isdir(path+fname+'%d' % suffix):
            suffix += 1
        fname = fname+'%d' % suffix
        os.makedirs(path+fname)
    return fname


def seconds2hms(seconds):
    """Convert seconds to hours, minutes, seconds."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s)
