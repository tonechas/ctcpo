#! /usr/bin/env python37
"""
Summary
-------
Colour texture classification through partial orders.

Extended summary
----------------
This script allows the user to perform colour texture classification 
on several texture datasets using different partial orders. It can be 
run via the command line or from the IDE.

The main functionality is comprised of feature extraction and 
classification. In addition, the program can automatically generate 
the LaTeX source code for creating tables with the results of the 
experiments as well as job scripts to be submitted to the queue 
manager of Finis Terrae II supercomputer (CESGA).

Examples
--------
`$ python ctcpo.py --action ef --dataset CBT --descriptor RankTransform`
    Extract RankTransform features from the images of CBT texture dataset 
    using the default parameters.
`$ python ctcpo.py --action c --argsfile settings_all.ini`
    Perform a texture classification experiment using the datasets and 
    descriptors defined in the file `settings_all.ini` and the classifiers 
    and parameters defined in the module `config.py`.
    
"""


#=========#
# IMPORTS #
#=========#

import copy
import IPython
import itertools
import os
import sys
import platform
import psutil
import textwrap
import traceback
import warnings

import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter
from skimage import io, color
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from unittest.mock import patch
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


import texdata
import hep
import reportex
import utils


#=========#
# CLASSES #
#=========#
 
class Configuration(object):
    """Class for initializing global variables."""
    def __init__(self, home=None, n_tests=5, n_folds=5, 
                 test_size=0.5, random_state=0, 
                 estimators=[(KNeighborsClassifier, dict(n_neighbors=[1]))]):
        self.home = (os.getcwd() if home is None else home)
        self.imgs = os.path.join(self.home, 'images')
        self.data = os.path.join(self.home, 'data')
        self.log = os.path.join(self.home, 'log')
        self.n_tests = n_tests
        self.n_folds = n_folds
        self.test_size = test_size
        self.random_state = random_state
        self.estimators = estimators


class MyArgumentParser(ArgumentParser):
    """Sub class of argparse.ArgumentParser."""
    def convert_arg_line_to_args(self, arg_line):
        """Allows inserting comments in the args file."""
        for arg in arg_line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg
    def _read_args_from_files(self, arg_strings):
        """Allows retaining names of files that contain arguments."""
        # Most of the following is copied from argparse.py
        # expand arguments referencing files
        new_arg_strings = []
        for arg_str in arg_strings:
            # for regular arguments, just add them back into the list
            if not arg_str or arg_str[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_str)
            # replace arguments referencing files with the file content
            else:
                try:
                    fn = arg_str[1:]
                    with open(fn) as args_file:
                        # Changed this: before was []
                        arg_strings = [fn]
                        for arg_line in args_file.read().splitlines():
                            for arg in self.convert_arg_line_to_args(arg_line):
                                arg_strings.append(arg)
                        arg_strings = self._read_args_from_files(arg_strings)
                        new_arg_strings.extend(arg_strings)
                except OSError:
                    # Also changed this: before was _sys.exc_info()[1]
                    err = sys.exc_info()[1]
                    self.error(str(err))
        # return the modified argument list
        return new_arg_strings
    
    
#===========#
# FUNCTIONS #
#===========#

def make_parser():
    """
    make_parser()
    
    Return a parser for command-line options.

    Returns
    -------
    parser : argparse.ArgumentParser
    
    """
    # Instantiate the argument parser
    first_line = 'CTCPO, a software utility for'
    second_line = 'Colour Texture Classification through Partial Orders'
    parser = MyArgumentParser(description=f'{first_line}\n{second_line}', 
                              allow_abbrev=True, 
                              fromfile_prefix_chars='@', 
                              formatter_class=RawTextHelpFormatter)

    # Specify which command-line options the program is willing to accept
    parser.add_argument(
        'argsfile', 
        nargs='?', 
        type=str, 
        help=textwrap.dedent('''\
                Name of the file (prefixed with a leading "@") from which 
                the arguments passed through the command line are read. If 
                used, it should be the first argument.'''))

    parser.add_argument(
        '--action', 
        type=str, 
        choices = ('ef', 'c', 'efc', 'j', 'l', 'df', 'dr', 'da'), 
        default='efc', 
        help=textwrap.dedent(
                '''\
                ACTION can be one of the following:
                    ef    Extract features
                    c     Classify
                    efc   Extract features and classify
                    j     Job (generate script)
                    l     Latex (generate LaTeX source code)
                    df    Delete features
                    dr    Delete results
                    da    Delete all (features and results)'''))

    parser.add_argument(
        '--dataset', 
        nargs='+', 
        type=str, 
        help='Names of folders with texture datasets')

    parser.add_argument(
        '--descriptor', 
        nargs='+', 
        type=str, 
        help='Names of image descriptors')

    parser.add_argument(
        '--order', 
        nargs='+', 
        type=str, 
        default=['linear'],
        help='Order used in rank features (default "linear")')

    parser.add_argument(
        '--radius', 
        nargs='+', 
        type=lambda s: [int(item) for item in s.split(',')], 
        default=[[1]], 
        help='Radii of local neighbourhoods, comma-separated (default "[1]"))')

    parser.add_argument(
        '--bands', 
        nargs='+', 
        type=str, 
        default=['RGB'],
        help='Priority of bands (default "RGB")')

    parser.add_argument(
        '--alpha', 
        nargs='+', 
        type=int, 
        default=[2],
        help='Divisor of the first channel (default "2")')

    parser.add_argument(
        '--cref', 
        nargs='+', 
        type=lambda s: [int(item) for item in s.split(',')], 
        #type=str,
        default=[0, 0, 0], 
        help='Reference color, comma-separated components (default "0,0,0")')

    parser.add_argument(
        '--seed', 
        nargs='+', 
        type=int, 
        default=[0], 
        help='Seed for the random colour orders (default "0")')

    parser.add_argument(
        '--partition', 
        type=str, 
        default='cola-corta',
        help='Partition of Finis Terrae-II cluster (default "cola-corta")')

    parser.add_argument(
        '--maxruntime', 
        type=float, 
        default=10,
        help='Maximum execution time in hours (default "10")')

    parser.add_argument(
        '--jobname', 
        type=str, 
        default='nonamejob.sh',
        help='Name of the job (default "nonamejob.sh")')

    return parser


def gen_datasets(folder, dataset_names):
    """
    gen_datasets(folder, dataset_names)
    
    Generator for texture datasets.

    Parameters
    ----------
    folder : str
        Full path of the folder where image datasets are stored.
    dataset_names : list of str
        List of dataset names.

    Yields
    ------
    texdata.TextureDataset
        Instance of the class `TextureDataset` defined in module `texdata`.
        
    """
    for name in dataset_names:
        yield getattr(texdata, name)(os.path.join(folder, name))


def gen_descriptors(args, maxdim=2**16):
    """
    gen_descriptors(args, maxdim=2**16)
    
    Generator for texture descriptors.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    maxdim : int, optional (default 2**16)
        Maximum dimensionality of the descriptor. This limit is intended 
        to avoid MemoryError when extracting features.
        
    Yields
    ------
    Instance of the class `HEP` defined in module `hep`.
    
    """
    def instantiate_hep(name, parameters):
        '''Instantiate a HEP descriptor.'''
        return getattr(hep, name)(**parameters)

    for descr, radius, order in itertools.product(
            args.descriptor, args.radius, args.order):
        extra_keys = []
        params = {'radius': radius, 'order': order}

        if order in ('lexicographic', 'bitmixing'):
            extra_keys.append('bands')
        elif order == 'alphamod':
            extra_keys.extend(['alpha', 'bands'])
        elif order == 'refcolor':
            extra_keys.append('cref')
        elif order == 'random':
            extra_keys.append('seed')

        if not extra_keys:
            obj = instantiate_hep(descr, params)
            if obj.dim <= maxdim:
                yield obj
        else:
            lists_of_values = [vars(args)[key] for key in extra_keys]
            for extra_values in itertools.product(*lists_of_values):
                for key, value in zip(extra_keys, extra_values):
                    params[key] = value
                obj = instantiate_hep(descr, params)
                if obj.dim <= maxdim:
                    yield obj

 
def apply_descriptor(dataset, descriptor, print_info=False):
    """
    apply_descriptor(dataset, descriptor, print_info=False)
    
    Compute the features of the given dataset using the given descriptor.

    Parameters
    ----------
    dataset : texdata.TextureDataset
        Object that encapsulates data of a texture dataset.
    descriptor : hep.HEP
        Object that encapsulates data of a texture descriptor.
    print_info : bool, optional (default `False`)
        If this is set to `True`, image path is displayed on
        the screen while features are being computed.

    Returns
    -------
    X : array
        Computed features. The number of rows is equal to the number of
        samples and the number of columns is equal to the dimensionality
        of the feature space. If an error occurs it returns `None`.
        
    """
    orders_for_gray = ('linear',)
    orders_for_uint8 = ('bitmixing', 'refcolor')
    dataset_id = dataset.acronym
    descriptor_id = descriptor.abbrev()
    order = descriptor.order
    n_samples = len(dataset.images)
    n_features = descriptor.dim

    try:
        X = np.zeros(shape=(n_samples, n_features), dtype=np.float64)
        for index, image_path in enumerate(dataset.images):
            if print_info:
                print(image_path, flush=True)
            img = io.imread(image_path)
            # Check that img is a colour image
            if (img.ndim == 3) and (3 <= img.shape[2] <= 4):
                # Remove the transparency layer if necessary
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                # Convert to grayscale if necessary
                if order in orders_for_gray:
                    img = color.rgb2gray(img)
                # Check that dtype of img is np.uint8
                if order in orders_for_uint8 and img.dtype != np.uint8:
                    print(f'Cannot compute {descriptor_id}.codemap() '
                          f'{image_path} has to be numpy.uint8')
                    raise TypeError
            # Check that img is a grayscale image and the order is compatible
            elif not (img.ndims == 2 and order in orders_for_gray):
                print(f'Cannot compute {descriptor_id}.codemap() '
                      f'{image_path} has to be a colour image')
                raise ValueError
            X[index] = descriptor(img)

    except Exception as ex:
        error_id = ex.__class__.__name__
        print(f'{error_id}: skipping {dataset_id}--{descriptor_id}')
        if error_id == 'MemoryError':
            print(psutil.virtual_memory(), flush=True)
            traceback.print_exc()
            sys.stdout.flush()
        X = None

    return X


def concatenate_feats(data_folder, dataset, descriptor):
    """Compute features through concatenation of texture models.

    Parameters
    ----------
    data_folder : str
        Full path of the folder where data are saved.
    dataset : texdata.TextureDataset
        Object that encapsulates data of a texture dataset.
    descriptor : hep.HEP
        Object that encapsulates data of a texture descriptor.
    
    Returns
    -------
    X : array
        Computed features. The number of rows is equal to the number of
        samples and the number of columns is equal to the sum of the 
        dimensionalities of the concatenated texture models. If an error 
        occurs in the call to `apply_descriptor`, it returns `None`.

    """
    dat_id = dataset.acronym
    params = {k: v for k, v in descriptor.__dict__.items()}
    feats = []

    for component in descriptor.components: 
        descr = component(**params)
        descr_id = descr.abbrev()
        feat_path = utils.filepath(data_folder, dat_id, descr_id)
        if os.path.isfile(feat_path):
            X = utils.load_object(feat_path)
        else:
            X = apply_descriptor(dataset, descr)
            if X is not None:
                utils.save_object(X, feat_path)
            else:
                break
        feats.append(X)
    else:
        X = np.concatenate(feats, axis=-1)

    return X


def extract_features(data_folder, imgs_folder, args):
    """"Compute texture features.
    
    Check whether features have been already computed. If they haven't, 
    extract features from each dataset using each descriptor in
    `args` and save them to disk. If the descriptor is multi-scale, 
    a separate file is created for each single value of the radius.

    Parameters
    ----------
    data_folder : string
        Full path of the folder where data are saved.
    imgs_folder : string
        Full path of the folder where texture datasets are stored.
    args : argparse.Namespace
        Command line arguments.
        
    """
    utils.boxed_text('Extracting features...', symbol='*')

    for dat in gen_datasets(imgs_folder, args.dataset):
        dat_id = dat.acronym
        for descr in gen_descriptors(args):
            for rad in descr.radius:
                descr_rad = copy.deepcopy(descr)
                descr_rad.radius = [rad]
                descr_rad_id = descr_rad.abbrev()
                feat_path = utils.filepath(data_folder, dat_id, descr_rad_id)
                if os.path.isfile(feat_path):
                    print(f'Found {dat_id}--{descr_rad_id}', flush=True)
                else:
                    print(f'Computing {dat_id}--{descr_rad_id}', flush=True)
                    if hasattr(descr_rad, 'components'):
                        X = concatenate_feats(data_folder, dat, descr_rad)
                    else:
                        X = apply_descriptor(dat, descr_rad)
                    if X is not None:
                        utils.save_object(X, feat_path)
                        del X


def get_features(folder, dataset, descriptor):
    """Return texture features for a single dataset and descriptor.

    Parameters
    ----------
    folder : string
        Full path of the folder where data are saved.
    dataset : texdata.TextureDataset
        Object that encapsulates data of a texture dataset.
    descriptor : hep.HEP
        Object that encapsulates data of a texture descriptor.

    Returns
    -------
    X : array
        Texture features. The number of rows is equal to the number of
        samples and the number of columns is equal to the dimensionality
        of the feature space. If an error occurs within the call to 
        `apply_descriptor`, returns None.
        
    """
    multiscale_features = []
    dataset_id = dataset.acronym
    for rad in descriptor.radius:
        descr_single = copy.deepcopy(descriptor)
        descr_single.radius = [rad]
        descr_single_id = descr_single.abbrev()
        feat_path = utils.filepath(folder, dataset_id, descr_single_id)
        if os.path.isfile(feat_path):
            X = utils.load_object(feat_path)
        else:
            print(f'Computing {dataset_id}--{descr_single_id}')

            if hasattr(descr_single, 'components'):
                X = concatenate_feats(folder, dataset, descr_single)
            else:
                X = apply_descriptor(dataset, descr_single)
            if X is not None:
                utils.save_object(X, feat_path)
            else:
                break
        multiscale_features.append(X)
    else:
        X = np.concatenate(multiscale_features, axis=-1)
    return X


def grid_search_cv(X, y, clf, param_grid, n_folds, test_size, random_state):
    """Tune hyper-parameters through grid search an cross-validation.

    Parameters
    ----------
    X : array
        Texture features (one row per image).
    y : array
        Class labels.
    clf : class
        Class that implements a classifier, for example `sklearn.svm.SVC`, 
        `sklearn.neighbors.KNeighborsClassifier`, etc.
    param_grid : dict
        Dictionary with parameters names (string) as keys and lists of 
        parameter settings as values, which defines the exhaustive search 
        to be performed by `GridSearchCV`.
    n_folds : int
        Number of folds used for cross-validation. Must be at least 2.
    test_size : float
        Proportion of samples used for testing. Value ranges from 0 to 1.
    random_state : int
        Seed for the random number generator. This affects the splits into 
        train and test, and the cross-validation folds.

    Returns
    -------
    X : array
        Computed features. The number of rows is equal to the number of
        samples and the number of columns is equal to the dimensionality
        of the feature space.
        
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    gscv = GridSearchCV(estimator=clf(),
                        cv=StratifiedKFold(n_splits=n_folds, 
                                           random_state=random_state),
                        param_grid=param_grid,
                        return_train_score=False)
    with warnings.catch_warnings():
        # !!! Context manager intended to turn off the following warning:
        # C:\Miniconda3\envs\python37\lib\site-packages\sklearn\
        # model_selection\_search.py:813: 
        # DeprecationWarning: The default of the `iid` parameter will 
        # change from True to False in version 0.22 and will be removed 
        # in 0.24. This will change numeric results when test-set 
        # sizes are unequal.
        # DeprecationWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        gscv.fit(X_train, y_train)
        
    best_clf = clf(**gscv.best_params_)
    best_clf.fit(X_train, y_train)
    test_score = best_clf.score(X_test, y_test)
    return [gscv, test_score]


def classify(data_folder, imgs_folder, args, estimators, test_size, n_tests, 
             n_folds, random_state):
    '''Compute classification results.
    
    Check whether features have been already classified. If not,
    perform classification using each estimator for each dataset and
    descriptor, and save results to disk.

    Parameters
    ----------
    data_folder : string
        Full path of the folder where data are saved.
    imgs_folder : string
        Full path of the folder where texture datasets are stored.
    args : argparse.Namespace
        Command line arguments.
    estimators : list of tuples
        Each tuple consist in a classifier (such as nearest neighbour,
        support vector machine, etc.) and the parameters used for
        optimization through `GridSearch`.
    test_size : float
        Proportion (between 0.0 and 1.0) of the dataset to
        include in the test split.
    n_tests : int
        Number of reshuffling and splitting operations.
    n_folds : int
        Number of folds used for cross-validation. Must be at least 2.
    random_state : int
        Seed for the random number generator. This affects the splits into 
        train and test, and the cross-validation folds.
        
    '''
    utils.boxed_text('Classifying...', symbol='*')

    for clf, param_grid in estimators:
        clf_id = ''.join(
                [letter for letter in clf.__name__ if letter.isupper()])
        for dat in gen_datasets(imgs_folder, args.dataset):
            dat_id = dat.acronym
            for descr in gen_descriptors(args):
                descr_id = descr.abbrev()                
                result_path = utils.filepath(
                        data_folder, dat_id, descr_id, clf_id)
                if os.path.isfile(result_path):
                    print(f'Loading {dat_id}--{descr_id}--{clf_id}', 
                          flush=True)
                    result = utils.load_object(result_path)
                else:
                    X = get_features(data_folder, dat, descr)
                    if X is None:
                        print(f'Skipping {dat_id}--{descr_id}--{clf_id}', 
                              flush=True)
                        continue
                    print(f'Computing {dat_id}--{descr_id}--{clf_id}', 
                          flush=True)
                    y = dat.labels
                    np.random.seed(random_state)
                    random_states = np.random.randint(size=n_tests, low=0, 
                                                      high=1000)    
                    # It is essential to pass a different `rstate` to 
                    # `grid_search_cv` for each grid search. Otherwise  
                    # data are split into train and test always the same  
                    # way and as a consequence, the results returned by 
                    # `grid_search_cv` are identical.
                    result = [grid_search_cv(X, y, clf, param_grid, n_folds, 
                                             test_size, rs) 
                              for rs in random_states]
                    utils.save_object(result, result_path)
        
                best_scores = [g.best_score_ for g, _ in result]
                test_scores = [ts for _, ts in result]
                print(f'Mean best cv score: {100*np.mean(best_scores):.2f}%')
                print(f'Mean test score: {100*np.mean(test_scores):.2f}%\n')


def delete_one_file(path_args):
    """Delete a single file"""
    fname = utils.filepath(*path_args)
    utils.attempt_to_delete_file(fname)


def delete_features(data_folder, imgs_folder, args):
    """Delete previously computed features.

    Search for the features files corresponding to the passed datasets,
    and descriptors, and delete them (if they exist).

    Parameters
    ----------
    data_folder : str
        Path of the folder where data are stored.
    imgs_folder : string
        Full path of the folder where texture datasets are stored.
    args : argparse.Namespace
        Command line arguments.
        
    """
    for dat in gen_datasets(imgs_folder, args.dataset):
        dat_id = dat.acronym
        for descr in gen_descriptors(args):
            descr_id = descr.abbrev()
            if len(descr.radius) == 1:
                delete_one_file([data_folder, dat_id, descr_id])
            else:
                for rad in descr.radius:
                    descr_single = copy.deepcopy(descr)
                    descr_single.radius = [rad]
                    descr_single_id = descr_single.abbrev()
                    delete_one_file([data_folder, dat_id, descr_single_id])


def delete_results(data_folder, imgs_folder, args, estimators):
    """Delete previously computed classification results.

    Search for the classification results files corresponding 
    to the passed datasets, descriptors and estimators, and delete them 
    (if they exist).

    Parameters
    ----------
    data_folder : str
        Path of the folder where data are stored.
    imgs_folder : string
        Full path of the folder where texture datasets are stored.
    args : argparse.Namespace
        Command line arguments.
    estimators : list of tuples, optional (default `None`)
        Estimators used to assess generalization error. List of tuples 
        of the form (classifier, parameters).
        
    """
    for clf, _ in estimators:
        caps = [letter for letter in clf.__name__  if letter.isupper()]
        clf_id = ''.join(caps)
        for dat in gen_datasets(imgs_folder, args.dataset):
            dat_id = dat.acronym
            for descr in gen_descriptors(args):
                descr_id = descr.abbrev()
                delete_one_file([data_folder, dat_id, descr_id, clf_id])
                if len(descr.radius) > 1:
                    for rad in descr.radius:
                        descr_single = copy.deepcopy(descr)
                        descr_single.radius = [rad]
                        descr_single_id = descr_single.abbrev()
                        delete_one_file([data_folder, dat_id, descr_single_id, 
                                         clf_id])


def confirm_deletion(outcome):
    '''Confirm deletion of files
    
    Parameters
    ----------
    outcome : str
        The type of data to be deleted, for example features or
        classification results.

    Returns
    -------
    True if the user answers affirmatively to the confirmation prompt, 
    False otherwise.

    '''
    utils.boxed_text('Deleting features...', symbol='*')

    ans = input(f'Are you sure you want to delete {outcome}? (Y/[N]) ')
    print()

    if ans and 'yes'.startswith(ans.lower()):
        print(f'Preparing to delete {outcome}...\n')
        return True
    else:
        print(f'No {outcome} deleted\n')
        return False


def display_arguments(args):
    """Print arguments on screen."""

    def show_key_values(key, values):
        """Print pairs of argument name and list of values."""
        if values:
            print(f'{key.capitalize()}:')
            for value in values:
                print(f'- {value}')

    print('SETTINGS\n' + 8*'-')
    if args.argsfile:
        print(f"Settings taken from:\n- {args.argsfile}")
    action = args.action.lower()
    print(f"Action:\n- {args.action}")
    if action in ('j', 'job'):
        print(f"Job name:\n- {args.jobname}")
        print(f"Partition:\n- {args.partition}")

    for key in ('dataset', 'descriptor', 'order', 'radius'):
        show_key_values(key, vars(args).get(key))
    if set.intersection(set(['lexicographic', 'bitmixing', 'alphamod']), 
                        set(args.order)):
        show_key_values('bands', args.bands)
    if 'alphamod' in args.order:
        show_key_values('alpha', args.alpha)
    if 'refcolor' in args.order:
        show_key_values('cref', args.cref)
    if 'random' in args.order:
        show_key_values('seed', args.seed)
    
    print('Classifiers')
    for clf, _ in config.estimators:
        print(f"- {clf.__name__}")
    

def parse_arguments():
    """Parse command-line options and arguments"""
    parser = make_parser()
    if len(sys.argv) == 1:
        # No command-line arguments, intended for running the program from IDE
        testargs = ['@args_all.txt', 
                    '--dataset', 'CBT', 
                    '--descriptor', 'LocalDirectionalRankCoding', 
                    '--action', 'j', 
                    '--radius', '1', '2',
                    '--order', 'linear', 'product', 'random',
                    '--seed', '0', '1', '2',
                    '--maxruntime', '2']
        #testargs = '--action c --dataset NewBarkTex --descriptor LocalConcaveConvexMicroStructurePatterns --radius 1 --order lexicographic --bands RGB'.split()
#        testargs = ('--act efc '
#                    '--dataset CBT NewBarkTex '
#                    '--desc ImprovedCenterSymmetricLocalBinaryPattern '
#                    'RankTransform LocalConcaVeMicroStructurePatterns ' 
#                    '--order linear product lexicographic bitmixing alphamod refcolor random '
#                    '--bands RGB RBG GRB BGR '
#                    '--radius 1 2 3 1,2 '
#                    '--cref 0,0,0 127,127,127 '
#                    '--alpha 2 4 '
#                    '--seed 0 1 2 3 4').split()
        fake_argv = [sys.argv[0]] + testargs
        with patch.object(sys, 'argv', fake_argv):
            args = parser.parse_args()
    elif len(sys.argv) == 2 and sys.argv[1][0] != '@':
        # This branch is necessary because when the script is executed
        # through GNU `parallel` and `srun` all the command-line arguments
        # are considered a single string.            
        srun_args = sys.argv[1].split()
        fake_argv = [sys.argv[0]] + srun_args
        with patch.object(sys, 'argv', fake_argv):
            args = parser.parse_args()
    else:
        args = parser.parse_args()

    # Abort execution if datasets or descriptors have not been passed in
    if not args.dataset or not args.descriptor:
        sys.exit('error: datasets or descriptors have not been passed in')
    
    return args


def job_script(dataset, descriptor, maxruntime, count):
    """
    !!! Missing docstring
    
    """
    if maxruntime < 0 or maxruntime > 100:
        print('Error: MAXRUNTIME must be in the range [0, 100]')
        sys.exit()
    
    partition = 'cola-corta' if maxruntime <= 10 else 'thin-shared '

    days = int(maxruntime//24)
    hours = int(maxruntime%24)
    minutes = int((maxruntime - 24*days - hours)*60)

    time = f'{hours:02}:{minutes}:00'
    if days > 1:
        time = f'{days}-{time}'
    else:
        time = f'{time}  '

    jobname = f'job{count:05}'
    
    job = ['#!/bin/sh', 
           '#SBATCH --mail-type=begin            # send email when the job begins',
           '#SBATCH --mail-type=end              # send email when the job ends',
           '#SBATCH --mail-user=antfdez@uvigo.es # e-mail address',
           '#SBATCH --mem=24GB                   # allocated memory',
           f'#SBATCH -p {partition}                # partition name',
           f'#SBATCH -t {time}                 # maximum execution time',
           f'#SBATCH -J {jobname}                  # name for the job']

    srun = ['srun python /home/uvi/dg/afa/ctcpo/ctcpo.py',
            '--action ef',
            f'--dataset {dataset.__class__.__name__}',
            f'--descriptor {descriptor.__class__.__name__}',
            f'--radius {descriptor.radius[0]}',
            f'--order {descriptor.order}']

    if descriptor.order in ('lexicographic', 'bitmixing', 'alphamod'):
        srun.append(f'--bands {descriptor.bands}')
        if descriptor.order == 'alphamod':
            srun.append(f'--alpha {descriptor.alpha}')
    elif descriptor.order == 'refcolor':
        cref = ','.join([format(i, '02x') for i in descriptor.cref]).upper()
        srun.append(f'--cref {cref}')
    elif descriptor.order == 'random':
        srun.append(f'--seed {descriptor.seed}')

    srun = ' '.join(srun)
    job.append(srun)

    script_fn = f'{jobname}.sh'
    if platform.system() == 'Linux':
        script_fn = os.path.join(
                '/mnt/netapp2/Store_uni/home/uvi/dg/afa/texture/jobs',
                script_fn)

    with open(script_fn, 'w') as f:
        print('\n'.join(job), file=f)
    print('\n'.join(job))


def generate_job_scripts(data_folder, imgs_folder, args):
    """"Create job scripts.
    
    Check whether features have been already computed. If they haven't, 
    create a job script for each dataset-descriptor pair.

    Parameters
    ----------
    data_folder : string
        Full path of the folder where data are saved.
    imgs_folder : string
        Full path of the folder where texture datasets are stored.
    args : argparse.Namespace
        Command line arguments.
        
    """
    print('Generating job scripts...\n')

    count = 0

    for dat in gen_datasets(imgs_folder, args.dataset):
        dat_id = dat.acronym
        for descr in gen_descriptors(args):
            for rad in descr.radius:
                descr_rad = copy.deepcopy(descr)
                descr_rad.radius = [rad]
                descr_rad_id = descr_rad.abbrev()
                feat_path = utils.filepath(data_folder, dat_id, descr_rad_id)

                if not os.path.isfile(feat_path):
                    count += 1
                    print(f'job{count:05}.sh >> {dat_id}--{descr_rad_id}', 
                          flush=True)
                    job_script(dat, descr_rad, args.maxruntime, count)


def main():
    pass


#%%============#
# MAIN PROGRAM #
#==============#
if __name__ == '__main__':
    
    # Print welcome message
    utils.boxed_text('COLOUR TEXTURE CLASSIFICATION THROUGH PARTIAL ORDERS')

    # Initialize global variables
    # `config` is a module or an instance of `Configuration`.
    try:
        import config
    except ImportError:
        print('config.py not found')
        print('Using default configuration')
        config = Configuration()
        
    args = parse_arguments()
        
    # Show settings
    display_arguments(args)
    
    # Make sure data directory exists
    IPython.utils.path.ensure_dir_exists(config.data)
    
    # Execute the proper action
    option = args.action.lower()
    if option == 'ef':
        extract_features(config.data, config.imgs, args)
    elif option == 'c':
        classify(config.data, config.imgs, args, config.estimators, 
                 config.test_size, config.n_tests, 
                 config.n_folds, config.random_state)
    elif option == 'efc':
        extract_features(config.data, config.imgs, args)
        classify(config.data, config.imgs, args, config.estimators, 
                 config.test_size, config.n_tests, 
                 config.n_folds, config.random_state)
    elif option == 'j':
        generate_job_scripts(config.data, config.imgs, args)
    elif option == 'l':
        reportex.generate_latex(args, config)
    elif option == 'df':
        if confirm_deletion('features'):
            delete_features(config.data, config.imgs, args)
    elif option == 'dr':
        if confirm_deletion('results'):
            delete_results(config.data, config.imgs, args, config.estimators)
    elif option == 'da':
        if confirm_deletion('features and results'):
            delete_features(config.data, config.imgs, args)
            delete_results(config.data, config.imgs, args, config.estimators)
    else:
        print(f"Argument \"--action {args.action}\" is not valid")
        print('Run:')
        print('    $ python tcpo.py --help')
        print('for help on command-line options and arguments')