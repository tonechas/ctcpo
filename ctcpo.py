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
experiments as well as job scripts to be submitted to the queue manager 
of Finis Terrae II supercomputer (CESGA).

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
import textwrap

import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter
from skimage import io
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from unittest.mock import patch
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


import texdata
import hep
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
        for arg_string in arg_strings:
            # for regular arguments, just add them back into the list
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)
            # replace arguments referencing files with the file content
            else:
                try:
                    fn = arg_string[1:]
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
        choices = ('ef', 'extract_features', 
                   'c', 'classify', 
                   'efc', 'extract_features&classify', 
                   'j', 'job', 
                   'l', 'latex', 
                   'df', 'delete_features', 
                   'dr', 'delete_results', 
                   'da', 'delete_all'), 
        default='efc', 
        help=textwrap.dedent(
                '''\
                ACTION can be one of the following:
                    ef    extract_features
                    c     classify
                    efc   extract_features&classify
                    j     job (generate script)
                    l     latex (generate LaTeX source code)
                    df    delete_features
                    dr    delete_results
                    da    delete_all (features and results)'''))

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
        '--jobname', 
        type=str, 
        default='job_noname.sh',
        help='Name of the job (default "nonamejob")')

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
        help='Radii of local neighbourhoods, comma-separated (default "1"))')

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
        default='thinnodes',
        help='Partition of Finis Terrae-II cluster (default "thinnodes")')

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


def gen_descriptors(args):
    """
    gen_descriptors(args)
    
    Generator for texture descriptors.

    Parameters
    ----------
    args : dict
        Dictionary with the descriptors and the different options 
        for the descriptors.

    Yields
    ------
    Instance of the class `HEP` defined in module `hep`.
    
    """
    def instantiate_hep(name, parameters):
        '''Instantiate a HEP descriptor.'''
        return getattr(hep, name)(**parameters)

    descriptors = args.descriptor
    radii = args.radius
    orders = args.order
    for descr, radius, order in itertools.product(descriptors, radii, orders):
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
            yield instantiate_hep(descr, params)
        else:
            lists_of_values = [vars(args)[key] for key in extra_keys]
            for extra_values in itertools.product(*lists_of_values):
                for key, value in zip(extra_keys, extra_values):
                    params[key] = value
                yield instantiate_hep(descr, params)

 
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
        of the feature space.
        
    """
    X = np.zeros(shape=(len(dataset.images), descriptor.dim), dtype=np.float64)
    for i, image_path in enumerate(dataset.images):
        if print_info:
            print(image_path, flush=True)
        img = io.imread(image_path)
        X[i] = descriptor(img)
    return X


def extract_features(folder, datasets, descriptors):
    """"
    extract_features(folder, datasets, descriptors)
    
    Compute texture features.
    
    Check whether features have been already computed. If not, extract
    features from each dataset in `datasets` using each descriptor in
    `descriptors` and save them to disk. If the descriptor is multi-scale, 
    a separate file is created for each single value of the radius.

    Parameters
    ----------
    folder : string
        Full path of the folder where data are saved.
    datasets : generator
        Generator of instances of `texdata.TextureDataset` (for example 
        `Kather`, `MondialMarmi20`, etc.) to extract features from.
    descriptors : generator
        Generator of instances of `hep.HEP` (for example `RankTransform`,
        `LocalDirectionalRankCoding`, etc.) used to compute features.
        
    """
    utils.boxed_text('Extracting features...', symbol='*')
    print(f'Setting up the datasets and descriptors...\n')

    for dat, descr in itertools.product(datasets, descriptors):
        dat_id = dat.acronym
        for rad in descr.radius:
            descr_single = copy.deepcopy(descr)
            descr_single.radius = [rad]
            descr_single_id = descr_single.abbrev()
            feat_path = utils.filepath(folder, dat_id, descr_single_id)
            print(f'Computing {dat_id}--{descr_single_id}', flush=True)
            if not os.path.isfile(feat_path):
                X = apply_descriptor(dat, descr_single, print_info=False)
                utils.save_object(X, feat_path)
    print()


def get_features(folder, dataset, descriptor):
    """
    get_features(folder, dataset, descriptor)
    
    Return texture features for a single dataset and descriptor.

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
        of the feature space.
        
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
            print(f'Computing features {dataset_id}--{descr_single_id}')
            X = apply_descriptor(dataset, descr_single, print_info=False)
            utils.save_object(X, feat_path)
        multiscale_features.append(X)
    else:
        X = np.concatenate(multiscale_features, axis=-1)
    return X


def grid_search_cv(X, y, clf, param_grid, n_folds, test_size, random_state):
    """
    grid_search_cv(X, y, clf, param_grid, n_folds, test_size, random_state)
    
    Tune hyper-parameters through grid search an cross-validation.

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
    gscv.fit(X_train, y_train)
    best_clf = clf(**gscv.best_params_)
    best_clf.fit(X_train, y_train)
    test_score = best_clf.score(X_test, y_test)
    return [gscv, test_score]


def classify(folder, datasets, descriptors, estimators, test_size, n_tests, 
             n_folds, random_state):
    '''
    classify(folder, datasets, descriptors, estimators, test_size, n_tests, 
             n_folds, random_state)
    
    Compute classification results.
    
    Check whether features have been already classified. If not,
    perform classification using each estimator for each dataset and
    descriptor, and save results to disk.

    Parameters
    ----------
    folder : string
        Full path of the folder where data are saved.
    datasets : generator
        Generator of instances of `texdata.TextureDataset` (for example 
        `Kather`, `MondialMarmi20`, etc.) to extract features from.
    descriptors : generator
        Generator of instances of `hep.HEP` (for example `RankTransform`,
        `LocalDirectionalRankCoding`, etc.) used to compute features.
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
    print(f'Setting up the datasets, descriptors and classifiers...\n')

    for items in itertools.product(datasets, descriptors, estimators):
        dat, descr, (clf, param_grid) = items
        dat_id = dat.acronym
        descr_id = descr.abbrev()
        clf_id = clf.__name__
        print(f'Computing {dat_id}--{descr_id}--{clf_id}', flush=True)
        result_path = utils.filepath(folder, dat_id, descr_id, clf.__name__)
        if os.path.isfile(result_path):
            result = utils.load_object(result_path)
        else:
            X = get_features(folder, dat, descr)
            y = dat.labels
            np.random.seed(random_state)
            random_states = np.random.randint(size=n_tests, low=0, high=1000)    
            # It is essential to pass a different `rstate` to 
            # `grid_search_cv` for each grid search. Otherwise data are 
            # split into train and test always the same way and as a 
            # consequence, the results returned by `grid_search_cv` 
            # are identical.
            result = [
                grid_search_cv(X, y, clf, param_grid, n_folds, test_size, rs)
                for rs in random_states]
            utils.save_object(result, result_path)

        best_scores = [g.best_score_ for g, _ in result]
        test_scores = [ts for _, ts in result]
        print(f'Mean best cv score: {100*np.mean(best_scores):.2f}%')
        print(f'Mean test score: {100*np.mean(test_scores):.2f}%\n')


def job_script(data, loops, job_id, partition):
    """
    !!!
    """
    print('Generating job script...')


def generate_latex(args):
    """
    !!!
    """
    print('Generating LaTeX...')


def delete_files(folder, datasets, descriptors, estimators=None, both=True):
    '''
    delete_files(folder, datasets, descriptors, estimators=None, both=True)
    
    Delete previously computed features or classification results.

    Search for the features/classification results files corresponding 
    to the passed datasets, descriptors and estimators, and delete them 
    (if they exist).

    Parameters
    ----------
    folder : str
        Path of the folder where data are stored.
    datasets : generator of `texdata.TextureDataset` instances
        Texture datasets.
    descriptors : generator of `hep.HEP` instances
        Image descriptors.
    estimators : list of tuples, optional (default `None`)
        Estimators used to assess generalization error. List of tuples 
        of the form (classifier, parameters).
    both : bool, optional (default `False`)
        If `True` features and classification data are deleted. If `False`, 
        either features or classification data are deleted (but not both).

    '''
    def attempt_to_delete_file(path):
        try:
            os.remove(path)
            print(f'Deleted {path}', flush=True)
        except FileNotFoundError:
            print(f'{path} could not be deleted', flush=True)

    if both:
        outcome = 'features and classification results'
    elif estimators is not None:
        outcome = 'classification results'
    else:
        outcome = 'features'
    ans = input(f'Are you sure you want to delete {outcome}? (Y/[N]) ')
    print()
    
    if ans and 'yes'.startswith(ans.lower()):
        print(f'Preparing to delete {outcome}...\n')
        for dat, descr in itertools.product(datasets, descriptors):
            dat_id = dat.acronym
            for rad in descr.radius:
                descr_single = copy.deepcopy(descr)
                descr_single.radius = [rad]
                descr_single_id = descr_single.abbrev()
                feat_args = [folder, dat_id, descr_single_id]
                if both or (estimators is None):
                     attempt_to_delete_file(utils.filepath(*feat_args))
                if both or (estimators is not None):
                    for clf, _ in estimators:
                        res_args = feat_args + [clf.__name__]
                        attempt_to_delete_file(utils.filepath(*res_args))
    else:
        print(f'No {outcome} deleted\n')
    print()


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
    print()
    

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
    
    # Parse command-line options and arguments
    parser = make_parser()
    if len(sys.argv) == 1:
        # No command-line arguments, intended for running the program from IDE
        #testargs = ['@args_one.txt']
        testargs = '--act j --dataset CBT --desc RankTransform'.split()
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
        
    # Show settings
    display_arguments(args)
    
    # Make sure data directory exists
    IPython.utils.path.ensure_dir_exists(config.data)

    # Set up the generators of datasets and descriptors
    datasets = gen_datasets(config.imgs, args.dataset)
    descriptors = gen_descriptors(args)
    
    # Execute the proper action
    option = args.action.lower()
    if option in ('ef', 'extract_features'):
        extract_features(config.data, datasets, descriptors)
    elif option in ('c', 'classify'):
        classify(config.data, datasets, descriptors, 
                 config.estimators, config.test_size, 
                 config.n_tests, config.n_folds, config.random_state)
    elif option in ('efc', 'extract_features&classify'):
        extract_features(config.data, datasets, descriptors)
        # The generators are now exhausted and need to be refreshed
        datasets = gen_datasets(config.imgs, args.dataset)
        descriptors = gen_descriptors(args)
        classify(config.data, datasets, descriptors, 
                 config.estimators, config.test_size, 
                 config.n_tests, config.n_folds, config.random_state)
    elif option in ('j', 'job'):
        job_script(config.data, None, args.jobname, args.partition)
    elif option in ('l', 'latex'):
        generate_latex(args)
    elif option in ('df', 'delete_features'):
        delete_files(config.data, datasets, descriptors, both=False)
    elif option in ('dr', 'delete_results'):
        delete_files(config.data, datasets, descriptors, 
                     config.estimators, both=False)
    elif option in ('da', 'delete_all'):
        delete_files(config.data, datasets, descriptors, 
                     config.estimators, both=True)
    else:
        print(f"Argument \"--action {args.action}\" is not valid")
        print('Run:')
        print('    $ python tcpo.py --help')
        print('for help on command-line options and arguments')