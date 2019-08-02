#! /usr/bin/env python37
"""
This module includes all the necessary stuff to automatically generate the 
LaTeX source code of the report with the results obtained from colour texture 
classification through partial orders.

"""


import os
import numpy as np

import config
import hep
import utils


#===========#
# CONSTANTS #
#===========#

dataset_ids = {'CBT': 'CBT', 
               'ForestMacro': 'ForestMacro', 
               'ForestMicro': 'ForestMicro', 
               'Kather': 'Kather', 
               'KTHTIPS2b': 'KTH2b', 
               'KylbergSintorn': 'KylSin', 
               'MondialMarmi20': 'Mond20', 
               'NewBarkTex': 'NewBarkTex', 
               'Outex13': 'Outex13', 
               'PapSmear': 'PapSmear', 
               'Parquet': 'Parquet', 
               'PlantLeaves': 'PlantLeaves', 
               'STex': 'STex', 
               'VxCTSG': 'VxCTSG'}

order_ids = {'linear': '', 
             'product': 'prod', 
             'lexicographic': 'lex', 
             'alphamod': 'alpha', 
             'bitmixing': 'mix', 
             'refcolor': 'ref', 
             'random': 'rand'}

#===========#
# FUNCTIONS #
#===========#

def beginning():
    """Generate the preamble of the report"""
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


def itemize_ttt(items):
    """Generate a list of items in tele type (typewriter or monospace) font
    
    Parameters
    ----------
    items : list of str
        List of items.
    
    Returns
    -------
    out : str
        LaTeX source code of the itemize environment.
        
    """
    itemstr = '\n'.join([rf"  \item \texttt{{{item}}}" for item in items])
    out =  (r"\begin{itemize}"
            f"\n{itemstr}\n"
            r"\end{itemize}")
    return out


def intro_args(args):
    """Generate section with command line arguments
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    """
    code = [r"\section*{Command line arguments: \texttt{args}}",
            "",
            r"\begin{itemize}",
            r"\item \texttt{dataset}",
            itemize_ttt(sorted(args.dataset)),
            r"\item \texttt{descriptor}",
            itemize_ttt(sorted(args.descriptor)),
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
    """Generate section with table of descriptor dimensionalities

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    """
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
    """Generate section thar describes the validation methods"""
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


def read_score(folder, dat_id, descr_id, clf_id):
    """Read test scores from a file and compute the average value

    Parameters
    ----------
    folder : string
        Full path of the folder where data are saved.
    dat_id : string
        Short name of a dataset.
    descr_id : string
        Short name of a descriptor.
    clf_id : string
        Short name of a classifier.
        
    Returns
    -------
    ts_avg : float
        Average of test scores.

    """
    result_path = utils.filepath(folder, dat_id, descr_id, clf_id)
    if os.path.isfile(result_path):
        result = utils.load_object(result_path)
        test_scores = [ts for _, ts in result]
        ts_avg = 100*np.mean(test_scores)
        return ts_avg
    else:
        return None


def get_scores(descr, clf, dat, rad, order, args):
    """Return the computed average test scores

    Parameters
    ----------
    descr : string
        Full name of a descriptor.
    clf : string
        Instance of a classifier.
    dat : string
        Full name of a dataset.
    rad : list of int
        Radii of the local neighbourhoods.
    args : argparse.Namespace
        Command line arguments.
        
    Returns
    -------
    rates : list of float
        Average of test scores. 
        - If data are not available, the list is empty. 
        - If the descriptor does not depend on additional parameters 
        the list contains a single value. 
        - If the descriptor depends on additional parameters, each 
        item on the list corresponds to a particular combination of 
        the parameter values.
        
    Raises
    ------
    ValueError if the value of parameter `order` is invalid .

    """
    def append_score(rates, folder, dat_id, descr_id, clf_id):
        rate = read_score(folder, dat_id, descr_id, clf_id)
        if rate is not None:
            rates.append(rate)
    
    dat_id = dataset_ids[dat]
    clf_id = utils.all_caps(clf.__name__)
    descr_short = utils.all_caps(descr)
    order_id = order_ids[order]

    rates = []
    if order in ['linear', 'product']:
        descr_id = f'{descr_short}{order_id}{rad}'.replace(' ', '')
        append_score(rates, config.data, dat_id, descr_id, clf_id)

    elif order in ['lexicographic', 'bitmixing']:
        for bands in args.bands:
            descr_id = f'{descr_short}{order_id}{bands}{rad}'.replace(' ', '')
            append_score(rates, config.data, dat_id, descr_id, clf_id)

    elif order in ['alphamod']:
        for bands in args.bands:
            for alpha in args.alpha:
                descr_id = f'{descr_short}{order_id}{alpha}{bands}{rad}'
                descr_id = descr_id.replace(' ', '')
                append_score(rates, config.data, dat_id, descr_id, clf_id)

    elif order in ['refcolor']:
        for rgb in args.cref:
            cref = ''.join([format(i, '02x') for i in rgb]).upper()
            descr_id = f'{descr_short}{order_id}{cref}{rad}'.replace(' ', '')
            append_score(rates, config.data, dat_id, descr_id, clf_id)

    elif order in ['random']:
        for seed in args.seed:
            descr_id = f'{descr_short}{order_id}{seed}{rad}'.replace(' ', '')
            append_score(rates, config.data, dat_id, descr_id, clf_id)

    else:
        raise ValueError('invalid order')

    return rates


def single_entry(value, highest, tol=1e-6):
    """Content (LaTeX source code) of a cell with a single score
    
    Parameters
    ----------
    value : float
        Average of test scores. If `value` is greater than or equal to 
        `highest`, is displayed in bold font.
    highest : float
        Highest value of the row.
    tol : float, optional (default 1e-6)
        Tolerance value used in the comparison of floats.

    """
    if abs(value - highest) < tol:
        return rf'\bfseries{{{value:.1f}}}'
    else:
        return rf'{value:.1f}'


def multi_entry(lst, highest, tol=1e-6):
    """Content (LaTeX source code) of a cell with a range of scores
    
    Parameters
    ----------
    lst : list of float
        Range of test scores in a cell. If any of the values in the range 
        is greater than or equal to `highest`, the whole range is displayed 
        in bold font.
    highest : float
        Highest value of the row.
    tol : float, optional (default 1e-6)
        Tolerance value used in the comparison of floats.

    """
    maxcell = max(lst)
    mincell = min(lst)
    if abs(maxcell - mincell) < 1e-6:
         return single_entry(maxcell, highest)
    elif abs(maxcell - highest) < tol:
        return rf'\bfseries{{{mincell:.1f}-{maxcell:.1f}}}'
    else:
        return rf'{mincell:.1f}-{maxcell:.1f}'
        

def get_cell(scores, highest):
    """Return the content (LaTeX code) of a single cell of a table
    
    Parameters
    ----------
    scores : list
        Score values.
    highest : float
        Highest score of a row of a table.

    """
    if not scores:
        return ''
    elif len(scores) == 1:
        return single_entry(scores[0], highest)
    else:
        return multi_entry(scores, highest)


def get_row(descr, clf, dat, rad, args):
    """Single row of a table"""
    cells = [get_scores(descr, clf, dat, rad, order, args) 
             for order in args.order]
    return cells

def row_not_empty(row):
    """Return True if the row has som values in it"""
    non_empty_cells = [cell for cell in row if cell]
    return len(non_empty_cells) > 0

def multirow(descr, clf, dat, args):
    """Rows of a table corresponding to the same dataset"""
    rows = {tuple(rad): get_row(descr, clf, dat, rad, args) 
                        for rad in args.radius}
    if not any(row_not_empty(row) for row in rows.items()):
        # There are no scores for this radius
        return ''
    else:
        # There are some scores for this radius
        first_row = True
        code = []
        for rad in rows.keys():
            if row_not_empty(rows[rad]):
                highest = max([max(cell) for cell in rows[rad] if cell])
                cells = [f'{get_cell(cell, highest)}' for cell in rows[rad]]
                src_row = ' & '.join(cells)
                if first_row:
                    col_heading = dat
                    first_row = False
                else:
                    col_heading = ''
                code.append(f"{col_heading} & {list(rad)} & {src_row} \\\\")
        return '\n'.join(code)


def table(descr, clf, args):
    """Generate a table with the results of a subsection"""
    code = [rf"\begin{{longtable}}{{ll{'r'*len(args.order)}}}",
            r"Dataset & Radius & "
            rf"{' & '.join([order.capitalize() for order in args.order])} \\",
            r"\hline"]
    for dat in sorted(args.dataset):
        code.append(multirow(descr, clf, dat, args))
    code.append("\\end{longtable}\n")
    return '\n'.join(code)


def subsection(descr, clf, args):
    """Generate the LaTeX source code of a subsection of the report"""
    code = [rf"\subsection*{{{clf.__name__}}}",
            table(descr, clf, args),
            r"\newpage",
            "",
            ""]
    return '\n'.join(code)


def section(descr, args):
    """Generate the LaTeX source code of a section of the report"""
    code = [rf"\section*{{{descr}}}"]
    for clf, _ in config.estimators:
        code.append(subsection(descr, clf, args))
    return '\n'.join(code)


def sections(args):
    """Iterate through the sections that make up the report"""
    sects = [section(descr, args) for descr in sorted(args.descriptor)]
    return '\n'.join(sects)


def generate_latex(args, config):
    """Automatically generate LaTeX source code for the report"""
    print('\nGenerating LaTeX code...\n')
        
    code = [beginning(),
            intro_args(args),
            intro_config(),
            intro_dims(args),
            intro_validation(),
            sections(args),
            r"\end{document}"]
    src = '\n'.join(code)
    print(src)
    with open(os.path.join(config.home, 'latex', 'report.tex'), 'w') as f:
        print(src, file=f)