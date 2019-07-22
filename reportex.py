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

def generate_latex_old(args):
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


def beginning():
    code = [r"\documentclass{{article}}",
            r"",
            r"\usepackage[a4paper,margin=20mm]{geometry}",
            r"\usepackage{longtable}",
            r"\usepackage{listings}",
            r"",
            r"\title{Color texture classification based on product order}",
            r"\author{Antonio Fern\'{a}ndez}",
            r"\date{\today}",
            r"",
            r"\begin{document}",
            r"",
            r"\maketitle",
            r""]
    return '\n'.join(code)


def itemize(items):
    itemstr = '\n'.join([rf"  \item \texttt{{{item}}}" for item in items])
    out =  (r"\begin{itemize}"
            f"\n{itemstr}\n"
            r"\end{itemize}")
    return out


def intro_args(args):
    """Generate section with command line arguments"""
    code = [r"\section*{Command line arguments: \texttt{args}}",
            "",
            r"\begin{itemize}",
            r"\item \texttt{dataset}",
            itemize(sorted(args.dataset)),
            r"\item \texttt{descriptor}",
            itemize(sorted(args.descriptor)),
            rf"\item \texttt{{order = {args.order}}}",
            rf"\item \texttt{{radius = {args.radius}}}",
            rf"\item \texttt{{bands = {args.bands}}}",
            rf"\item \texttt{{alpha = {args.alpha}}}",
            rf"\item \texttt{{cref = {args.cref}}}",
            rf"\item \texttt{{seed = {args.seed}}}",
            r"\end{itemize}",
            ""]
    
    return '\n'.join(code)


def intro_config():
    """Generate section with configuration settings"""
    filename = "config.py"
    #filepath = os.path.join(os.getcwd(), filename)
    code = [rf"\section*{{Settings: \texttt{{{filename}}}}}",
            "",
            r"\lstinputlisting[language=Python, firstline=40, lastline=48]"
            f"{{{filename}}}",
            ""]
    
    return '\n'.join(code)


def intro_dims(args):
    """Generate section with table of descriptor dimensionalities"""
    code = [r'\section*{Dimensionalities}',
            '',
            r'\begin{longtable}{llrr}',
            r'Descriptor & Radius & Canonical order & Product order \\',
            r'\hline']
    
    for descr in sorted(set(args.descriptor)):
        for i, radius in enumerate(args.radius):
            acron = ''.join([letter for letter in descr if letter.isupper()])
            lin = getattr(hep, descr)(radius=radius, order='linear').dim
            prod = getattr(hep, descr)(radius=radius, order='product').dim
            line = rf'{acron if i == 0 else ""} & {radius} & {lin} & {prod} \\'
            code.append(line)

    code.append(r'\end{longtable}')
    code.append('')
    
    return '\n'.join(code)


def intro_validation():
    code = [r"\section*{Validation methods}",
            "",
            r"\subsection*{\texttt{StratifiedShuffleSplit}}",
            "",
            r"Data are randomly split \texttt{{n\_tests}} times "
            "into train and test sets.",
            "",
            r"\subsection*{\texttt{StratifiedKFold}}",
            "",
            r"The hyperparameters of the classifiers are tuned through cross-"
            r"validation. Each train set is split into \texttt{{n\_folds}} "
            r"folds. The combination of hyperparameters that yields the "
            r"highest score is determined by \texttt{GridSearchCV}. "
            r"Each classifier is fitted \texttt{n\_tests} times with a "
            r"training set and the best parameters, and then is evaluated "
            r"on the corresponding test set. Generalization error is "
            r"estimated by averaging the resulting \texttt{n\_tests} scores.",
            "",
            r"\newpage",
            ""]
    
    return '\n'.join(code)

#        vals = []
#        for o in osect:
#        #for o in hep._orders:
#            # `same`: list of descriptors with the same name, radius and order
#            same = [d for d in imdescr if d.__class__.__name__ == s and d.radius == r and d.order == o]
#            if not same:
#                #print('Not required:  {}--{}--{}--{}--{}'.format(db, s, r, o, clf.__name__))
#                vals.append(None)
#            elif len(same) == 1:
#                result_path = utils.filepath(config.DATA, db, same[0], clf)
#                if os.path.isfile(result_path):
#                    #print('Reading single:  ', result_path)
#                    result = utils.load_object(result_path)
#                    acc = 100*np.mean([ts for g, ts in result])
#                    vals.append(acc)
#                else:
#                    print('Not found (single):  ', result_path)
#                    vals.append(None)
#            else:
#                accs = []
#                for descr in same:
#                    result_path = utils.filepath(config.DATA, db, descr, clf)
#                    if os.path.isfile(result_path):
#                        #print('Reading multiple:  ', result_path)
#                        result = utils.load_object(result_path)
#                        accs.append(100*np.mean([ts for g, ts in result]))
#                    else:
#                        print('Not found (multi):  ', result_path)
#                if not accs:
#                    vals.append(None)
#                elif len(accs) == 1:
#                    vals.append(accs[0])
#                elif len(accs) > 1:
#                    vals.append([np.min(accs), np.max(accs)])

def row():
    return "*" #!!!


def multirow(dat, est, descr, args, config):
    code = []
    for i, rad in enumerate(args.radius):
        first = dat if i == 0 else ''
        code.append(f"{first} & {rad} {row()}")
    return '\n'.join(code)
        

def table(dat, est, args, config):
    code = [rf"\begin{{longtable}}{{ll{'r'*len(args.order)}}}",
            r"Dataset & Radius & "
            rf"{' & '.join([order.capitalize() for order in args.order])} \\",
            r"\hline"]
    for descr in args.descriptor:
        code.append(multirow(dat, est, descr, args, config))
    code.append("\\end{longtable}\n")
    return '\n'.join(code)


def subsection(dat, est, args, config):
    clf, _ = est
    code = [rf"\subsection*{{{clf.__name__}}}",
            table(dat, est, args, config),
            r"\newpage",
            "",
            ""]
    return '\n'.join(code)


def section(dat, args, config):
    code = [rf"\section*{{{dat}}}"]
    for est in config.estimators:
        code.append(subsection(dat, est, args, config))
    return '\n'.join(code)


def sections(args, config):
    sects = [section(dat, args, config) for dat in sorted(args.dataset)]
    return '\n'.join(sects)


def generate_latex(args, config):
    """Automatically generate LaTeX source code for report"""
    print('\nGenerating LaTeX code...')
        
    code = [beginning(),
            intro_args(args),
            intro_config(),
            intro_dims(args),
            intro_validation(),
            sections(),
            r"\end{document}"]
    src = '\n'.join(code)
    print(src)
    with open(os.path.join(config.data, 'report.tex'), 'w') as f:
        print(src, file=f)