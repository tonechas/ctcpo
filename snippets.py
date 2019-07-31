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



#########################################################################
# hep.py

d = CompletedLocalBinaryCountSM(order='alphamod', radius=[1])
img = np.random.randint(low=0, high=256, size=(5, 7, 3), dtype=np.uint8)
d = LocalConcaveConvexMicroStructurePatterns(order='bitmixing')
#img = np.uint8(np.arange(12).reshape((3, 4)))
#img = np.array([[0, 1, 2, 3], [4, 5.3, 6.9, 7], [8, 9, 10, 11]])
img = np.arange(4*5*3).reshape((4,5,3))
h = d(img)
print(np.nonzero(h))
np.random.seed(0)
z = np.random.permutation(256)


#class Drexel(TextureDataset):
#    r"""Class for the Drexel dataset.
#
#    The Drexel Texture Database includes stationary colour textures
#    representing 20 different materials such as bark, carpet, knit,
#    sponge, etc. The dataset features 1560 images per class which are the
#    result of combining 30 viewing conditions (generated by varying the
#    object-camera distance and the viewpoint) and 52 illumination directions.
#
#    Image format : .png (RGB)
#    Sample size : 85-299 x  71-291 px (range of nrows x range of ncols)
#                  147x75 - 299x139 px (range of number of pixels)
#
#    Examples
#    --------
#    Sample file name : aluminium_foil\sample_d\15d-scale_2_im_1_col.png
#        class : aluminium_foil
#        corresponding CURET sample number: 15
#        sample : d
#        scale number : 2
#        image number : 1 =>
#            object pose : frontal
#            illumination direction: frontal
#
#    References
#    ----------
#    .. [1] https://www.cs.drexel.edu/~kon/codeanddata/texture/index.html
#    """
#
#    def __init__(self, dirpath):
#        super().__init__(dirpath)
#        self.scales = np.asarray([self.get_scale(img) for img in self.images])
#        self.viewpoints = np.asarray([self.get_viewpoint(img) for img in self.images])
#        self.illuminations = np.asarray([self.get_illumination(img) for img in self.images])
#        self.acronym = 'KTH2b'
#            
#
#    def get_class(self, img):
#        """Extract the class label from the given image file name."""
#        subfolder = os.path.dirname(img)
#        folder, _ = os.path.split(subfolder)
#        return os.path.split(folder)[-1]
#
#
#    def get_scale(self, img):
#        """Extract the scale number from the given image file name."""
#        _, filename = os.path.split(img)
#        head, tail = filename.split('-')
#        return int(tail.split('_')[1])
#
#
#    def get_viewpoint(self, img):
#        """Extract the viewpoint from the given image file name."""
#        _, filename = os.path.split(img)
#        head, tail = filename.split('-')
#        img_number = tail.split('_')[3]
#        if img_number in ['1','2','3','10']:
#            return 'Frontal'
#        elif img_number in ['4','5','6','11']:
#            return '22.5 Right'
#        elif img_number in ['7','8','9','12']:
#            return '22.5 Left'
#        else:
#            raise ValueError('Incorrect image number')
#
#            
#    def get_illumination(self, img):
#        """Extract the illumination direction from the given image file name."""        
#        _, filename = os.path.split(img)
#        head, tail = filename.split('-')
#        img_number = tail.split('_')[3]
#        if img_number in ['1','4','7']:
#            return 'Frontal'
#        elif img_number in ['2','5','8']:
#            return '45 top'
#        elif img_number in ['3','6','9']:
#            return '45 side'
#        elif img_number in ['10','11','12']:
#            return 'Ambient'
#        else:
#            raise ValueError('Incorrect image number') 


#class RawFooT(TextureDataset):
#    """Implementation of the RawFooT dataset
#                
#    Images info:
#            Comprehends 68 classes(46 samples per class) of raw food and grains
#            such as corn, chicken breast, pomegranate, salmon and tuna. 
#            The materials were acquired under 46 different illumination 
#            conditions resulting in as many image samples for each class. 
#
#            Extension  : .png
#            Resolution : 800x800
#            shape      : 800x800x3
#                
#            An example of img name :  0007-01
#                                        class : 007
#                                        sample : 01
#                                        
#    Remarks: 
#       -In this dataset we need to crop images.
#    """
#    
#    def __init__(self, imgdir):
#        
#        super().__init__(imgdir)
#        
#        self.labels = []                
#        
#        for img in self.images:
#            for i in range(4):
#                self.labels.append(self.get_label(img))
#        self.labels = np.asarray(self.labels)
#        self.label = 'RawFooT'
#            
#    
#    ## Returns the class of the img
#    def get_class(self, img):
#        """Returns the class of the given img."""
#        
#        head, tail = os.path.split(img)
#        head = head.split('\\')
#        return head[-1]


#    from IPython.utils.path import ensure_dir_exists
#    
#    if platform.system() == 'Linux':
#        home = r'/mnt/netapp2/Store_uni/home/uvi/dg/afa/texture'
#    elif platform.system() == 'Windows':
#        home = r'C:\texture'
#    
#    imgs = os.path.join(home, 'images')
#    data = os.path.join(home, 'data')
#    log = os.path.join(data, 'log')
#    
#    ensure_dir_exists(data)
#    destination = os.path.join(data, 'KylbergSintorn')
#    ensure_dir_exists(destination)
#    download_KylbergSintorn(destination, x=4, y=4)

def job_script(folder, job_id, partition, datasets, descriptors, estimators=None)#data, loops):
    """
    !!!
    """
    for dat, descr in itertools.product(datasets, descriptors):
        dat_id = dat.acronym
        for rad in descr.radius:
            descr_single = copy.deepcopy(descr)
            descr_single.radius = [rad]
            descr_single_id = descr_single.abbrev()
            feat_args = [folder, dat_id, descr_single_id]
            if estimators is None:
                this_one = utils.filepath(*feat_args)
            else:
                for clf, _ in estimators:
                    res_args = feat_args + [clf.__name__]
                    this_one = utils.filepath(*res_args)



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
    print('Generating job script...')


def command_line_arguments(folder, datasets, descriptors, estimators=None):
    ''' !!! Missing docstring.'''
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



#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#

#def lexicographic_order(neighbour, central, bandperm=(0, 1, 2), comp=np.less):
#    """
#    Compare a peripheral pixel and the central pixel of the neighbourhood
#    using the lexicographic order.
#
#    Parameters
#    ----------
#    neighbour : array
#        Subimage corresponding to a peripheral pixel of the neighbourhood.
#    central : array
#        Subimage corresponding to the central pixels of the neighbourhood.
#    bandperm : tuple
#        Permutation of the chromatic channels (defined by their indices)
#        which establishes the priority considered in the lexicographic order.
#    comp : Numpy function, optional (default np.less)
#        Comparison function (np.greater, np.less_equal, etc.).
#
#    Returns
#    -------
#    result : boolean array
#        Truth value of comp (neighbour, central) element-wise according to
#        the lexicographic order.
#
#    References
#    ----------
#    .. [1] E. Aptoula and S. Lefêvre
#           A comparative study on multivariate mathematical morphology
#           https://doi.org/10.1016/j.patcog.2007.02.004
#    """
#    weights = np.asarray([256**np.arange(central.shape[2])[::-1]])
#    ord_central = np.sum(central[:, :, bandperm]*weights, axis=-1)
#    ord_neighbour = np.sum(neighbour[:, :, bandperm]*weights, axis=-1)
#    result = comp(ord_neighbour, ord_central)
#    return result
#
#
#def bitmixing_order(neighbour, central, lut, bandperm=(0, 1, 2), comp=np.less):
#    """
#    Compare a peripheral pixel and the central pixel of the neighbourhood
#    using the bit mixing order.
#
#    Parameters
#    ----------
#    neighbour : array
#        Subimage corresponding to a peripheral pixel of the neighbourhood.
#    central : array
#        Subimage corresponding to the central pixels of the neighbourhood.
#    lut : array
#        3D Lookup table
#    bandperm : tuple
#        Permutation of the chromatic channels (defined by their indices)
#        which establishes the priority considered in the bit mixing order.
#    comp : Numpy function, optional (default np.less)
#        Comparison function (np.greater, np.less_equal, etc.).
#
#    Returns
#    -------
#    result : boolean array
#        Truth value of comp (neighbour, central) element-wise according to
#        the bit mixing product order.
#
#    References
#    ----------
#    .. [1] J. Chanussot and P. Lambert
#           Bit mixing paradigm for multivalued morphological filters
#           https://doi.org/10.1049/cp:19971007
#    """
#    ord_central = lut[tuple(central[:, :, bandperm].T)].T
#    ord_neighbour = lut[tuple(neighbour[:, :, bandperm].T)].T
#    result = comp(ord_neighbour, ord_central)
#    return result
#
#
#def refcolor_order(neighbour, central, cref=[0, 0, 0], comp=np.less):
#    """
#    Compare a peripheral pixel and the central pixel of the neighbourhood
#    using the reference color order.
#
#    Parameters
#    ----------
#    neighbour : array
#        Subimage corresponding to a peripheral pixel of the neighbourhood.
#    central : array
#        Subimage corresponding to the central pixels of the neighbourhood.
#    cref : tuple
#        Permutation of the chromatic channels (defined by their indices)
#        which establishes the priority considered in the lexicographic order.
#    comp : Numpy function, optional (default np.less)
#        Comparison function (np.greater, np.less_equal, etc.).
#
#    Returns
#    -------
#    result : boolean array
#        Truth value of comp (neighbour, central) element-wise according to
#        the reference color order.
#
#    .. [1] A. Ledoux, R. Noël, A.-S. Capelle-Laizé and C. Fernandez-Maloigne
#           Toward a complete inclusion of the vector information in
#           morphological computation of texture features for color images
#           https://doi.org/10.1007/978-3-319-07998-1_25
#    """
#    cref = np.asarray(cref).astype(np.int_)
#    dist_central = np.linalg.norm(central - cref)
#    dist_neighbour = np.linalg.norm(neighbour - cref)
#    result =  comp(dist_neighbour, dist_central)
#    return result

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

class Concatenation(object):
    """Class for concatenation of HEP descriptors."""
    def __init__(self, *descriptors):
        self.descriptors = descriptors

    def __call__(self, img):
        return np.concatenate([d(img) for d in self.descriptors])

    def __str__(self):
        return '+'.join([d.__str__() for d in self.descriptors])


