#! /usr/bin/env python37

"""!!!
"""


import os
import numpy as np

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


def intro_dims(args):
    """Generate section with table of descriptor dimensionalities"""
    code = []
    code.append(r'\section*{Dimensionalities}')
    code.append('')
    code.append(r'\begin{longtable}{llrr}')
    code.append(r'Descriptor & Radius & Canonical order & Product order \\')
    code.append(r'\hline')
    for descr in sorted(set(args.descriptor)):
        for i, radius in enumerate(args.radius):
            acron = ''.join([letter for letter in descr if letter.isupper()])
            lin = getattr(hep, descr)(radius=radius, order='linear').dim
            prod = getattr(hep, descr)(radius=radius, order='product').dim
            line = rf'{acron if i == 0 else ""} & {radius} & {lin} & {prod} \\'
            code.append(line)

    code.append(r'\end{longtable}')
    code.append('\n')
    
    return '\n'.join(code)


def intro_args(args):
    """Generate section with command line arguments"""
    code = []
    code.append(r'\section*{Command line arguments}')
    code.append('')
    code.append(r'\begin{itemize}')
    code.append(rf"\item \texttt{{order = {args.order}}}")
    code.append(rf"\item \texttt{{radius = {args.radius}}}")
    code.append(rf"\item \texttt{{bands = {args.bands}}}")
    code.append(rf"\item \texttt{{alpha = {args.alpha}}}")
    code.append(rf"\item \texttt{{cref = {args.cref}}}")
    code.append(rf"\item \texttt{{seed = {args.seed}}}")
    code.append(r'\end{itemize}')
    code.append(r'\newpage')
    code.append('')
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
    dbtex, imdescr = args, args#load_settings(args, config.IMGS)

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

