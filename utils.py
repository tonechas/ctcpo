"""Utility functions for Texture Classification through Partial Orders.
!!!
"""

import os
import pickle


def boxed_text(txt, symbol='#'):
    """Display a text enclosed on a rectangle whose lines are 
    made up of `symbol`.

    Parameters
    ----------
    txt : str
        Text to be displayed.
    symbol : str
        Symbol used to print the lines of the rectangle.
        
    Examples
    --------
    >>> boxed_text('Hello')
    #########
    # Hello #
    #########
    <BLANKLINE>
    >>> boxed_text('HELLO', '*')
    *********
    * HELLO *
    *********
    <BLANKLINE>
    """
    length = len(txt) + 4
    hline = symbol*length
    print(hline)
    print(f'{symbol} {txt} {symbol}')
    print(hline)
    print()


def display_sequence(seq, heading, symbol='-'):
    """Display the items of a sequence.

    Parameters
    ----------
    seq : A sequence such as list or tuple
        Sequence of elements, typically datasets, descriptors or classifiers.
    heading : str
        Title of the listing.
    symbol : str
        Symbol used to print horizontal lines.
        
    """
    length = len(heading)
    print(f'{heading}\n{symbol*length}')
    for item in seq:
        print(item)
    print()


def load_object(path):
    """Load an object from disk using pickle.

    Parameters
    ----------
    path : str
        Full path of the file where the object is stored.

    Returns
    -------
    obj : any type
        Object loaded from disk.
    """
    with open(path, 'rb') as fid:
        obj = pickle.load(fid)
    return obj


def save_object(obj, path):
    """Save an object to disk using pickle.

    Parameters
    ----------
    obj : any type
        Object to be saved.
    path : str
        Full path of the file where the object will be stored.
    """
    with open(path, 'wb') as fid:
        pickle.dump(obj, fid, 0)


def filepath(folder, *args, ext='pkl'):
    """Returns the full path of the file with the calculated results
    for the given dataset, descriptor, descriptor of the given dataset

    Parameters
    ----------
    folder : string
        Full path of the folder where results are saved.
    args : list or string
        Acronyms of instances of `TextureDataset`, `HEP`, 
        `KNeighborsClassifier`, etc.
    ext : string, optional
        File extension (default pkl).

    Returns
    -------
    fullpath : string
        The complete path of the file where features corresponding to the
        given dataset and descriptor (and estimator) are stored.
    
    Examples
    --------
    On macOS
        filepath('/Users/me/texture/data', 'Outex13', 'RTlexRGB')
    would return 
        '/Users/me/texture/data/Outex13--RTlexRGB.pkl'
    
    On Linux 
        filepath('/home/me/texture/data', 'STex', 'LCXNSP', ext='out')
    would return 
        '/home/me/texture/data/STex--LCXNSP.out'
    
    On Windows 
        filepath('C:\\Users\\me\\texture\\data', 'PapSmear', 'ICSLBP')
    would return 
        'C:\\Users\\me\\texture\\data\\PapSmear--ICSLBP.pkl'
    
    """
    filename = '--'.join(args) + '.' + ext
    fullpath = os.path.join(folder, filename)
    return fullpath


def attempt_to_delete_file(path):
    """Delete a file
    
    Parameters
    ----------
    path : str
        Full path of the file to be deleted.
        
    Notes
    -----
    If the file does not exist, an informative message is printed.
    
    """
    try:
        os.remove(path)
        print(f'Deleted {path}', flush=True)
    except FileNotFoundError:
        print(f'{path} could not be deleted', flush=True)


def subimage(img, pixel, radius):
    """Return the subimage used to vectorize the comparison between a given
    pixel and the central pixel of the neighbourhood.

    Parameters
    ----------
    img : array
        Input image.
    pixel : int
        Index of the peripheral pixel.
    radius : int
        Radius of the neighbourhood.

    Returns
    -------
    cropped : array
        Image crop corresponding to the given pixel and radius.

    Notes
    -----
    Neighbourhood pixels are numbered as follows:
                                                   R=3
                        R=2
      R=1                              23  22  21  20  19  18  17
                15  14  13  12  11      0   .   .   .   .   .  16
    7  6  5      0   .   .   .  10      1   .   .   .   .   .  15
    0  c  4      1   .   c   .   9      2   .   .   c   .   .  14
    1  2  3      2   .   .   .   8      3   .   .   .   .   .  13
                 3   4   5   6   7      4   .   .   .   .   .  12
                                        5   6   7   8   9  10  11

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(49).reshape(7, 7)
    >>> x
    array([[ 0,  1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, 10, 11, 12, 13],
           [14, 15, 16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25, 26, 27],
           [28, 29, 30, 31, 32, 33, 34],
           [35, 36, 37, 38, 39, 40, 41],
           [42, 43, 44, 45, 46, 47, 48]])
    >>> subimage(x, 2, 1)
    array([[15, 16, 17, 18, 19],
           [22, 23, 24, 25, 26],
           [29, 30, 31, 32, 33],
           [36, 37, 38, 39, 40],
           [43, 44, 45, 46, 47]])
    >>> subimage(x, 8, 2)
    array([[25, 26, 27],
           [32, 33, 34],
           [39, 40, 41]])
    >>> subimage(x, 22, 3)
    array([[1]])
    """
    diam = 2*radius + 1
    total = 4*(diam - 1)
    rows, cols = img.shape[:2]
    if pixel == total - 1:
        limits = 0, -2*radius, 0, -2*radius
    elif pixel in range(total - diam + 1, total - 1):
        limits = 0, -2*radius, total - pixel - 1, -2*radius + total - pixel - 1
    elif pixel == total - diam:
        limits = 0, -2*radius, 2*radius, cols
    elif pixel in range(total - 2*diam + 2, total - diam):
        limits = (total - diam - pixel,
                  total - diam - pixel - 2*radius, 2*radius, cols)
    elif pixel == total - 2*diam + 1:
        limits = 2*radius, rows, 2*radius, cols
    elif pixel in range(diam - 1, total - 2*diam + 1):
        limits = 2*radius, rows, pixel - diam + 2, pixel - diam + 2 - 2*radius
    elif pixel == diam - 2:
        limits = 2*radius, rows, 0, -2*radius
    elif pixel in range(diam - 2):
        limits = pixel + 1, pixel + 1 - 2*radius, 0, -2*radius
    else:
        raise ValueError('Invalid pixel')
    top, down, left, right = limits
    if img.ndim == 2:
        cropped = img[top: down, left: right]
    elif img.ndim == 3:
        cropped = img[top: down, left: right, :]
    else:
        raise ValueError('Invalid image')
    return cropped


if __name__ == "__main__":
    import doctest
    doctest.testmod()
