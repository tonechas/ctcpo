#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Texture classification through partial orders.

 using Finis Terrae II (CESGA).
 """


import os
import itertools
#import inspect
#import copy
import numpy as np
#from time import ctime
#from sklearn.model_selection import StratifiedShuffleSplit


import config
import hep
import utils


#==========
# FUNCTIONS
#==========


def introduction_validation(data, n_tests, test_size, n_folds):
    fpath = os.path.join(data, 'latex_introduction.txt')
    with open(fpath, 'r') as flatex:
        code = flatex.read().format(
            config_n_tests=n_tests,
            config_test_size=100*test_size,
            config_n_folds=n_folds)
    return code

def introduction_classifiers(estimators):
    code = []
    code.append('')
    code.append(r'\section*{Classifiers}')

    for clf, params in estimators:
        code.append('')
        code.append(r'\subsection*{{\texttt{{{0}}}}}'.format(clf.__name__))
        code.append('')
        code.append(r'\begin{itemize}')
        for k, v in params.items():
            if clf.__name__ == 'KNeighborsClassifier':
                line = r'\item \texttt{{{0}}}={1}'.format(k, v)
                code.append(line.replace(r'_', r'\_'))
            elif clf.__name__ == 'SVC':
                lst = ['2^{{{0}}}'.format(int(n)) for n in np.log2(v)]
                seq = '$[' + ', '.join(lst) +']$'
                line = r'\item \texttt{{{0}}}={1}'.format(k, seq)
                code.append(line.replace(r'_', r'\_'))
        code.append(r'\end{itemize}')
    code.append(r'\newpage')
    code.append('\n')

    return '\n'.join(code)

def introduction_dimensions(imdescr):
    names = sorted(set(d.__class__.__name__ for d in imdescr))
    code = []
    code.append(r'\section*{Dimensionalities}')
    code.append('')
    code.append(r'\begin{longtable}{llrr}')
    code.append(r'Descriptor & Radius & Canonical order & Product order \\')
    code.append(r'\hline')
    for d in names:
        for i, r in enumerate([[1], [2], [3], [1, 2], [1, 2, 3]]):
            d1 = getattr(hep, d)(radius=r, order='linear')
            d2 = getattr(hep, d)(radius=r, order='product')
            line = r'{0} & {1} & {2} & {3} \\'.format(
                d if i == 0 else '', r, d1.dim, d2.dim)
            code.append(line)
    code.append(r'\end{longtable}')
    code.append('\n')
    
    return '\n'.join(code)


def introduction_parameters():
    # !!! Remove hardcoding
    code = []
    code.append(r'\section*{Parameters}')
    code.append('')
    code.append(r'\begin{itemize}')
    code.append(
        r"\item \texttt{{bands = 'RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR'}}")
    code.append(r"\item \texttt{{cref = [0, 0, 0], [127, 127, 127]}}")
    code.append(r'\item \texttt{{alpha = 2, 4, 8, 16, 32}}')
    code.append(r'\item \texttt{{seed = 0, 1, 2, 3, 4}}')
    code.append(r'\end{itemize}')
    code.append(r'\newpage')
    code.append('\n')
    return '\n'.join(code)


def generate_latex(args):
    ''' !!! Missing docstring.'''

    def get_max_val(lst):
        ''' !!! Missing docstring.'''
        flattened = []
        for item in lst:
            if isinstance(item, (int, float)):
                flattened.append(item)
            elif isinstance(item, list):
                flattened.extend(item)
        return max(flattened) if flattened else None

    # Load settings
    dbtex, imdescr = load_settings(args, config.IMGS)

    # Display information
    utils.display_sequence(dbtex, 'Datasets', symbol='-')
    utils.display_sequence(imdescr, 'Descriptors', symbol='-')
    utils.display_sequence(
        [est[0].__name__ for est in config.ESTIMATORS],
        'Classifiers', symbol='-')
    utils.display_message('Generating LaTeX code', symbol='*')

    # Generate introductory sections
    code = introduction_validation(
        config.DATA, config.N_TESTS, config.TEST_SIZE, config.N_FOLDS)
    code += introduction_classifiers(config.ESTIMATORS)
    code += introduction_dimensions(imdescr)
    code += introduction_parameters()
    code = [code]


    # `sects`: names of the used descriptors (sorted alphabetically)
    sections = sorted(set(d.__class__.__name__ for d in imdescr))

#    # !!! Refactoring required
    for s in sections:
        # `s`: section title (is a descriptor name)
        code.append(r'\section*{{{0}}}'.format(s))
        tups = [tuple(d.radius) for d in imdescr if d.__class__.__name__ == s]
        # `rlst`: radii considered for descriptor `s`
        rlst = [*map(list, sorted(set(tups), key=lambda x: (len(x), x)))]
        # `osect`: orders considered for descriptor `s`
        #osect = sorted(set(d.order for d in imdescr if d.__class__.__name__ == s), key=lambda x: len(x))
        osect = hep._orders
        for k, (clf, params) in enumerate(config.ESTIMATORS):
            if k > 0:
                code.append(r'\newpage')
            code.append(r'\subsection*{{{0}}}'.format(clf.__name__))
            code.append(r'\begin{{longtable}}{{ll{0}}}'.format('r'*len(osect)))
            heading = r' & '.join([o.capitalize() for o in osect])
            code.append(r'Dataset & Radius & {0} \\'.format(heading))
            code.append(r'\hline')
            for db in dbtex:
                for i, r in enumerate(rlst):
                    if i == 0:
                        line = r'{0} & {1} '.format(db, r)
                    else:
                        line = r' & {0} '.format(r)
                    vals = []
                    for o in osect:
                    #for o in hep._orders:
                        # `same`: list of descriptors with the same name, radius and order
                        same = [d for d in imdescr if d.__class__.__name__ == s and d.radius == r and d.order == o]
                        if not same:
                            #print('Not required:  {}--{}--{}--{}--{}'.format(db, s, r, o, clf.__name__))
                            vals.append(None)
                        elif len(same) == 1:
                            result_path = utils.filepath(config.DATA, db, same[0], clf)
                            if os.path.isfile(result_path):
                                #print('Reading single:  ', result_path)
                                result = utils.load_object(result_path)
                                acc = 100*np.mean([ts for g, ts in result])
                                vals.append(acc)
                            else:
                                print('Not found (single):  ', result_path)
                                vals.append(None)
                        else:
                            accs = []
                            for descr in same:
                                result_path = utils.filepath(config.DATA, db, descr, clf)
                                if os.path.isfile(result_path):
                                    #print('Reading multiple:  ', result_path)
                                    result = utils.load_object(result_path)
                                    accs.append(100*np.mean([ts for g, ts in result]))
                                else:
                                    print('Not found (multi):  ', result_path)
                            if not accs:
                                vals.append(None)
                            elif len(accs) == 1:
                                vals.append(accs[0])
                            elif len(accs) > 1:
                                vals.append([np.min(accs), np.max(accs)])
                    for v in vals:
                        maxval = get_max_val(vals)
                        if v is None:
                            line += r'& '
                        elif isinstance(v, (int, float)):
                            if v == maxval:
                                line += r'& \bfseries{{{0:.1f}}} '.format(v)
                            else:
                                line += r'& {0:.1f} '.format(v)
                        elif isinstance(v, list):
                            if v[1] == maxval:
                                line += r'& \bfseries{{{0:.1f}--{1:.1f}}} '.format(v[0], v[1])
                            else:
                                line += r'& {0:.1f}--{1:.1f} '.format(v[0], v[1])
                    line += r'\\'
                    code.append(line)

            code.append(r'\end{longtable}')

        code.append(r'\newpage')
    code.append(r'\end{document}')

    latex_path = os.path.join(config.DATA, 'results.tex')
    with open(latex_path, 'w') as fid:
        fid.write('\n'.join(code))

def save_args_file(fargs, lines):
    ''' !!! Missing docstring.'''
    with open(fargs, 'w') as fid:
        for task in lines:
            print(task, file=fid)
    utils.dos2unix(fargs)


def save_job_script(fjob, commands):
    ''' !!! Missing docstring.'''
    with open(fjob, 'w') as fid:
        print(commands, file=fid)
    utils.dos2unix(fjob)


def command_line_arguments(data, loops):
    ''' !!! Missing docstring.'''
    datasets, descriptors, estimators = loops
    lines = []

    for dbase, descr in itertools.product(datasets, descriptors):
        for (clf, _) in estimators:
            if not (check_features(data, dbase, descr) \
                    and check_results(data, dbase, descr, clf)):
                task = '--action all --datasets {} '.format(dbase)
                task += '--descriptor {} '.format(
                    descr.__class__.__name__)
                task += '--order {} '.format(descr.order)
                task += '--radius {} '.format(
                    ' '.join([str(r) for r in descr.radius]))
                if descr.order in ('lexicographic', 'bitmixing'):
                    task += '--bands {} '.format(descr.bands)
                elif descr.order == 'alphamod':
                    task += '--alpha {} '.format(descr.alpha)
                elif descr.order == 'refcolor':
                    task += '--cref {} '.format(
                        ' '.join([str(i) for i in descr.cref]))
                elif descr.order == 'random':
                    task += '--seed {} '.format(descr.seed)
                lines.append(task)
                break
    return lines


def job_script(data, loops, job_id, partition):
    ''' !!! Missing docstring.'''
    datasets, descriptors, estimators = loops
    utils.display_sequence(datasets, 'Datasets', symbol='-')
    utils.display_sequence(descriptors, 'Descriptors', symbol='-')
    utils.display_sequence(
        [est[0].__name__ for est in estimators], 'Classifiers', symbol='-')

    utils.display_message('Creating arguments file and job script', symbol='*')

    tasks = command_line_arguments(data, loops)
    count = len(tasks)

    if count == 0:
        print('All tasks are already completed. Did not generate script.\n')
    else:
        args_file = 'args_{}.config'.format(job_id)
        save_args_file(args_file, tasks)
        job_file = 'job_{}.sh'.format(job_id)
        if partition == 'cola-corta':
            time_limit = '10:00:00'
            max_ntasks = 48
        elif partition == 'thinnodes':
            time_limit = '4-00:00:00'
            max_ntasks = 48
        elif partition == 'fatsandy':
            partition = 'fatsandy --qos shared'
            max_ntasks = 32
            time_limit = '4-04:00:00'

        # Workaround to keep the number of nodes low
        if count > max_ntasks:
            count = max_ntasks

        with open('job_template.sh', 'r') as fscript:
            script = fscript.read().format(
                jobname=job_id, partition=partition,
                maxtime=time_limit, ntasks=count)
        save_job_script(job_file, script)


def main():
    pass

#%%===========
# MAIN PROGRAM
#=============
if __name__ == '__main__':

    main()

#print('Mean test score: {}'.format(gscv.cv_results_['mean_test_score']))
#print('Best score : {}'.format(gscv.best_score_))
#print('Best params : {}'.format(gscv.best_params_))
#print('Score on test : {}\n\n'.format(test_score))

#print('parser.args = {}\n'.format(args))
#print('params = {}\n'.format(params))
#print('datasets = {}'.format(datasets))
#print('sys.argv = {}\n'.format(sys.argv))
#img = io.imread(r'C:\texture\images\MondialMarmi20\00\AcquaMarina_A_00_01.bmp')

#if not os.path.exists(log):
#    os.makedirs(log)
#log_file = os.path.join(log, 'log ' + datetime.datetime.now().__str__().split('.')[0].replace(':', '.') + '.txt')
#log = open(log_file, 'w')


#%%#####################
#### Fragmentos de utils 
########################
"""Utility functions for texture classification.
"""

import doctest
import unittest



def dos2unix(filename):
    """Replaces the DOS linebreaks (\r\n) by UNIX linebreaks (\n)."""
    with open(filename, 'r') as fid:
        content = fid.read()

    #text.replace(b'\r\n', b'\n')

    with open(filename, 'w', newline='\n') as fid:
        fid.write(content)


def hardcoded_subimage(img, pixel, radius):
    """Utility function for testing `subimage`.

    Parameters
    ----------
    img : array
        Input image (can be single-channnel or multi-channel).
    pixel : int
        Index of the peripheral pixel.
    radius : int
        Radius of the local neighbourhood.

    Returns
    -------
    subimg : array
        Subimage used to vectorize the comparison between a given pixel
        and the central pixel of the neighbourhood.
    """
    hardcoded_limits = {1: {0: (1, -1, 0, -2),
                            1: (2, None, 0, -2),
                            2: (2, None, 1, -1),
                            3: (2, None, 2, None),
                            4: (1, -1, 2, None),
                            5: (0, -2, 2, None),
                            6: (0, -2, 1, -1),
                            7: (0, -2, 0, -2),
                           },
                        2: {0: (1, -3, 0, -4),
                            1: (2, -2, 0, -4),
                            2: (3, -1, 0, -4),
                            3: (4, None, 0, -4),
                            4: (4, None, 1, -3),
                            5: (4, None, 2, -2),
                            6: (4, None, 3, -1),
                            7: (4, None, 4, None),
                            8: (3, -1, 4, None),
                            9: (2, -2, 4, None),
                            10: (1, -3, 4, None),
                            11: (0, -4, 4, None),
                            12: (0, -4, 3, -1),
                            13: (0, -4, 2, -2),
                            14: (0, -4, 1, -3),
                            15: (0, -4, 0, -4),
                           },
                        3: {0: (1, -5, 0, -6),
                            1: (2, -4, 0, -6),
                            2: (3, -3, 0, -6),
                            3: (4, -2, 0, -6),
                            4: (5, -1, 0, -6),
                            5: (6, None, 0, -6),
                            6: (6, None, 1, -5),
                            7: (6, None, 2, -4),
                            8: (6, None, 3, -3),
                            9: (6, None, 4, -2),
                            10: (6, None, 5, -1),
                            11: (6, None, 6, None),
                            12: (5, -1, 6, None),
                            13: (4, -2, 6, None),
                            14: (3, -3, 6, None),
                            15: (2, -4, 6, None),
                            16: (1, -5, 6, None),
                            17: (0, -6, 6, None),
                            18: (0, -6, 5, -1),
                            19: (0, -6, 4, -2),
                            20: (0, -6, 3, -3),
                            21: (0, -6, 2, -4),
                            22: (0, -6, 1, -5),
                            23: (0, -6, 0, -6),
                           },
                       }
    try:
        top, down, left, right = hardcoded_limits[radius][pixel]
        if 1 < img.ndim <= 2:
            subimg = img[top:down, left:right]
        elif img.ndim == 3:
            subimg = img[top:down, left:right, :]
        return subimg
    except (KeyError, NameError):
        print('No unit test available for this radius/pixel pair')
        raise


class TestSubimage(unittest.TestCase):
    """Test class for `subimage`."""

    def test_subimage(self):
        """Tests for `subimage`."""
        self.maxDiff = None
        np.random.seed(0)
        rows, cols = np.random.randint(low=7, high=15, size=2)
        gray = np.random.randint(0, high=255, size=(rows, cols))
        rgb = np.random.randint(0, high=255, size=(rows, cols, 3))
        for img in [gray, rgb]:
            for radius in range(1, 4):
                diam = 2*radius + 1
                total = 4*(diam - 1)
                for pix in range(total):
                    got = subimage(img, pix, radius)
                    expected = hardcoded_subimage(img, pix, radius)
                    self.assertSequenceEqual(got.tolist(), expected.tolist())


if __name__ == "__main__":
    doctest.testmod()
    unittest.main()
        
#%%##################################
#### Pruebas rápidas con iris dataset 
#####################################
    
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
clf, param_grid = config.estimators[0]
n_folds = config.n_folds
n_tests = config.n_tests
test_size = config.test_size
seed = config.seed
np.random.seed(seed)
random_states = np.random.randint(size=n_tests, low=0, high=1000)
result = [grid_search_cv(X, y, clf, param_grid, n_folds, rs) 
          for rs in random_states]

for d in gen_descriptors(args): print(d.abbrev())
for d, r, o in itertools.product(args['descriptor'], args['radius'], args['order']): print(f'{d}--{o}--{r}')



#############################
import yaml


def load_settings(filename):
    """Load datasets and descriptors settings from a file.

    Parameters
    ----------
    filename : str
        Name of the file which stores the settings.

    Returns
    -------
    settings : dict
        Dictionary with the datasets, descriptors and different options 
        for the descriptors.
        
    """
    try:
        with open(filename, 'r') as fid:
            settings = yaml.load(fid)
        return settings
    except FileNotFoundError:
        sys.exit(f'File not found: {filename}')


    # dict-like view of the attributes of `arguments`
    args = vars(arguments)
       
    # Read the datasets and descriptor settings from file
    if args['argsfile']:
        settings = load_settings(args['argsfile'])
        args = {**args, **settings}

