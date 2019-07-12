#! /usr/bin/env python37

"""Methods for characterizing texture images through
Histogram of Equivalent Patterns (HEP).
"""


import numpy as np

import utils


def square_frame_size(radius):
    """Compute the number of pixels of the external frame of square
    neighbourhoods of the given radii.

    Parameters
    ----------
    radius : list of int
        Radii of the local neighborhoods.

    Returns
    ----------
    points : list of int
        Number of pixels of the local neighborhoods.

    Examples
    --------
    >>> d = RankTransform(radius=[3])
    >>> square_frame_size(d.radius)
    [24]
    >>> square_frame_size([1, 2, 3])
    [8, 16, 24]

    """
    points = [8*r for r in radius]
    return points


def histogram(codes, nbins):
    """Compute the histogram of a map of pattern codes (feature values).

    Parameters
    ----------
    codes : array, dtype=int
        Array of feature values.
        For LCCMSP `codes` is a multidimensional array. It has two layers,
        one for the concave patterns and another for the convex patterns.
    nbins : int
        Number of bins of the computed histogram, i.e. number of
        possible different feature values.

    Returns
    -------
    h_norm : array
        Normalized histogram.

    """
    hist = np.bincount(codes.ravel(), minlength=nbins)
    h_norm = hist/hist.sum()
    return h_norm

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
_orders = ('linear',
           'product',
           'lexicographic',
           'alphamod',
           'bitmixing',
           'refcolor',
           'random')


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

class HEP(object):
    """Superclass for histogram of equivalent patterns descriptors.

    Notes
    -----
    `order='bitmixing'`
    In this case the triplet of unsigned integers
    [r7r6...r0, g7g6...g0, b7b6...1b0] is converted into an unsigned integer
    r7g7b7r6g6b6...r0g0b0 which is used as the ordering criterion.
    It is important to note that the description above corresponds to the
    default priority (`bands='RGB'`). For other band priority a permutation
    is applied.

    `self.lut` is a look-up table of shape `(256, 256, 256)` which is
    useful to efficiently compute the rank values.

    Examples
    --------
    >>> d = RankTransform(order='bitmixing')
    >>> d.lut[0, 0, 1]
    1
    >>> d.lut[0, 0, 2]
    8
    >>> d.lut[0, 1, 0]
    2
    >>> d.lut[0, 4, 0]
    128
    >>> d.lut[1, 0, 0]
    4
    >>> d.lut[8, 0, 0]
    2048

    References
    ----------
    .. [1] A. Fernandez, M. X. Alvarez, and F. Bianconi
           Texture Description through Histograms of Equivalent Patterns
           http://dx.doi.org/10.1007/s10851-012-0349-8

    """
    def __init__(self, **kwargs):
        """Initializer of a HEP instance.

        Parameters
        ----------
        kwargs : dictionary
            Keyword arguments listed below:
        order : string
            Order relation used in comparisons.
                'linear':
                    Canonical order for grayscale intensities.
                'product':
                    Product order (is a partial order).
                'lexicographic':
                    Lexicographic order based on priorities between the
                    chromatic components.
                'alphamod':
                    Lexicographic order with the first component
                    divided by alpha.
                'bitmixing':
                    Lexicographic order based on binary representation 
                    of intensities.
                'refcolor':
                    Preorder that relies on the Euclidean distance between
                    a color and a reference color.
                'random':
                    Order based on a random permutation of the
                    lexicographic order.
        radius : list of int, default is [1]
            Radii of the local neighborhoods.
        bands : str, default is 'RGB'
            Color band priorities.
        cref : list of int, default is [0, 0, 0]
            RGB coordinates of the reference color.
        seed : int, default is 0
            Seed for the pseudo-random number generator.

        """
        self.radius = kwargs.get('radius', [1])
        self.order = kwargs.get('order', 'linear')
        self.points = square_frame_size(self.radius)
        self.dims = self.compute_dims(self.points)
        self.dim = sum(self.dims)

        if self.order in ('lexicographic', 'bitmixing', 'alphamod'):
            self.bands = kwargs.get('bands', 'RGB')
            self.perm = tuple(['RGB'.index(b) for b in self.bands])

        if self.order == 'alphamod':
            self.alpha = kwargs.get('alpha', 2)

        if self.order == 'bitmixing':
            #Generate the lookup table for computing bit mixing order.
            bits = 8*np.dtype(np.uint8).itemsize
            levels = 2**bits
            channels = 3
            indices = np.arange(levels).astype(np.uint8)[:, None]

            exponents = [channels*np.arange(bits)[::-1] + ind
                         for ind in range(channels)[::-1]]

            weights = [2**exp for exp in exponents]

            red = np.sum(np.unpackbits(indices, axis=-1)*weights[0], axis=-1)
            green = np.sum(np.unpackbits(indices, axis=-1)*weights[1], axis=-1)
            blue = np.sum(np.unpackbits(indices, axis=-1)*weights[2], axis=-1)

            self.lut = red[:, None, None] \
                       + green[None, :, None] + blue[None, None, :]

        if self.order == 'random':
            self.seed = kwargs.get('seed', 0)
            np.random.seed(self.seed)
            bits = 8*np.dtype(np.uint8).itemsize
            levels = 2**bits
            channels = 3
            size = tuple(levels for i in range(channels))
            self.lut = np.random.permutation(levels**channels).reshape(size)

        if self.order == 'refcolor':
            self.cref = kwargs.get('cref', [0, 0, 0])


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        raise NotImplementedError("Subclasses should implement this!")


    def codemap(self, img, radius, points):
        """Return a map of feature values."""
        raise NotImplementedError("Subclasses should implement this!")


    def print_order_not_supported(self):
        order = self.order
        classname = self.__class__.__name__
        msg = f'{order} order is not supported for {classname} descriptor'
        raise ValueError(msg)

    def __call__(self, img):
        """Compute the feature vector of an image through a HEP descriptor.

        Parameters
        ----------
        img : array
            Input image.

        Returns
        -------
        hist : array
            Feature vector (histogram of equivalent patterns).
            
        """
        # Remove the transparency layer if necessary
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if len(self.radius) == 1:
            # Single scale
            codes = self.codemap(img, self.radius[0], self.points[0])
            hist = histogram(codes, self.dim)
        else:
            # Multi scale
            lst = []
            for rad in self.radius:
                single_scale_descr = self.get_single_scale(rad)
                lst.append(single_scale_descr(img))
            hist = np.hstack(lst)
        return hist


    def __str__(self):
        """"Return a string representation of the descriptor."""
        name = self.__class__.__name__
        order = self.order
        params = f"order='{order}', radius={self.radius}"
        if order in ('lexicographic', 'bitmixing', 'alphamod'):
            params += f", bands='{self.bands}'"
        if order == 'refcolor':
            params += f", cref={self.cref}"
        elif order == 'random':
            params += f", seed={self.seed}"
        elif order == 'alphamod':
            params += f", alpha='{self.alpha}'"
        return f"{name}({params})"


    def abbrev(self):
        """Return the abbreviated descriptor name.
        
        Parameters
        ----------
        descriptor : HEP.hep
            Instance of a HEP texture descriptor.
            
        Returns
        -------
        out : str
            Abbreviated name of the descriptor which is used to generate 
            the names of the files where the corresponding data (features 
            or/and classification results) are saved.
            
        Examples
        --------
        >>> RankTransform(order='linear', radius=[1]).abbrev()
        'RT[1]'
        >>> RankTransform(order='product', radius=[1, 2]).abbrev()
        'RTprod[1,2]'
        >>> params1 = dict(order='bitmixing', bands='GRB', radius=[1, 2, 3])
        >>> CompletedLocalBinaryCountSM(**params1).abbrev()
        'CLBCSMmixGRB[1,2,3]'
        >>> params2 = dict(radius=[1], alpha=2, order='alphamod', bands='BRG')
        >>> ImprovedCenterSymmetricLocalBinaryPattern(**params2).abbrev()
        'ICSLBPalpha2BRG[1]'
        
        """
        name = self.__class__.__name__
    
        short_descr = ''.join([letter for letter in name if letter.isupper()])
        order = self.order
        if order == 'linear':
            short_order = ''
        elif order == 'product':
            short_order = 'prod'
        elif order == 'lexicographic':
            short_order = 'lex'
        elif order == 'alphamod':
            short_order = 'alpha'
        elif order == 'bitmixing':
            short_order = 'mix'
        elif order == 'refcolor':
            short_order = 'ref'
        elif order == 'random':
            short_order = 'rand'
        else:
            raise ValueError('invalid order')
    
        if order in ('lexicographic', 'bitmixing'):
            suffix = self.bands
        elif order == 'alphamod':
            suffix = f"{self.alpha}{self.bands}"
        elif order == 'refcolor':
            suffix = ''.join([format(i, '02x') for i in self.cref]).upper()
        elif order == 'random':
            suffix = self.seed
        else:
            suffix = ''
        out = f"{short_descr}{short_order}{suffix}{self.radius}"
        return out.replace(' ', '')


    def get_single_scale(self, radius):
        """Create a single scale instance of the descriptor.

        Parameters
        ----------
        radius : int
            Radius of the neighbourhood.

        Returns
        -------
        descr : `HEP`
            Instance with the same attributes as `self` except `radius`,
            which contains a single scale.
            
        """
        params = {k: v for k, v in self.__dict__.items()}
        params['radius'] = [radius]
        descr = self.__class__(**params)
        return descr


    def compare(self, xarr, yarr, comp=np.less):
        """
        Compare two images according to a given order using the specified
        comparison operator.

        Parameters
        ----------
        xarr, yarr : arrays
            Images to be compared.
        comp : Numpy function, optional (default `np.less`)
            Comparison function (`np.greater`, `np.less_equal`, etc.).

        Returns
        -------
        result : boolean array
            Truth value of `comp(neighbour, central)` element-wise according
            to `order`.

        Examples
        --------
        >>> i1 = np.asarray([[0, 1, 2],
        ...                  [3, 4, 5]])
        ...
        >>> i2 = np.asarray([[0, 2, 1],
        ...                  [4, 1, 0]])
        ...
        >>> d_lin = RankTransform(order='linear')
        >>> np.testing.assert_array_equal(d_lin.compare(i1, i2, np.less_equal),
        ...                               [[ True,  True, False],
        ...                                [ True, False, False]])
        ...
        >>> a1 = np.asarray([[[0, 1, 2], [0, 1, 2]],
        ...                  [[3, 4, 5], [3, 4, 5]]])
        ...
        >>> a2 = np.asarray([[[2, 2, 2], [3, 3, 3]],
        ...                  [[3, 3, 3], [2, 2, 2]]])
        ...
        >>> d_prod = RankTransform(order='product')
        >>> np.testing.assert_array_equal(d_prod.compare(a1, a2, np.greater),
        ...                               [[False, False],
        ...                                [False,  True]])
        ...
        >>> b1 = np.asarray([[[0, 1, 2], [0, 1, 2]],
        ...                  [[3, 4, 5], [6, 7, 8]]])
        ...
        >>> b2 = np.asarray([[[2, 0, 1], [0, 1, 2]],
        ...                  [[4, 5, 6], [8, 8, 8]]])
        ...
        >>> d_lexrgb = RankTransform(order='lexicographic', bands='RBG')
        >>> np.testing.assert_array_equal(d_lexrgb.compare(b1, b2),
        ...                               [[ True, False],
        ...                                [ True,  True]])
        ...
        >>> d_lexbgr = RankTransform(order='lexicographic', bands='BGR')
        >>> np.testing.assert_array_equal(d_lexbgr.compare(b1, b2, np.greater),
        ...                               [[ True, False],
        ...                                [False, False]])
        ...
        >>> c1 = np.asarray([[[0b00000001, 0b00000010, 0b00000100]],
        ...                  [[0b11110000, 0b00111100, 0b00001111]],
        ...                  [[0b10000000, 0b11000000, 0b11100000]]])
        ...
        >>> c2 = np.asarray([[[0b00000001, 0b00000011, 0b00000111]],
        ...                  [[0b11110000, 0b00111100, 0b00001111]],
        ...                  [[0b10000000, 0b10000000, 0b10000000]]])
        ...
        >>> d_mixrgb = RankTransform(order='bitmixing', bands='RGB')
        >>> d_mixrgb.compare(c1, c2, np.less)
        array([[ True],
               [False],
               [False]])
        >>> d_mixrgb.compare(c1, c2, np.greater_equal)
        array([[False],
               [ True],
               [ True]])
        >>> d_mixbgr = RankTransform(order='bitmixing', bands='BGR')
        >>> d_mixbgr.compare(c1, c2, np.less)
        array([[ True],
               [False],
               [False]])
        >>> z1 = np.asarray([[[0, 0, 1], [0, 3, 4]],
        ...                  [[3, 4, 0], [1, 1, 1]]])
        ...
        >>> z2 = np.asarray([[[0, 0, 1], [5, 0, 0]],
        ...                  [[1, 1, 1], [0, 3, 0]]])
        ...
        >>> d_ref0 = RankTransform(order='refcolor', cref=[0, 0, 0])
        >>> d_ref0.compare(z1, z2, np.less)
        array([[False, False],
               [False,  True]])
        >>> d_ref0.compare(z1, z2, np.greater_equal)
        array([[ True,  True],
               [ True, False]])
        >>> d_ref9 = RankTransform(order='refcolor', cref=[9, 9, 9])
        >>> d_ref9.compare(z1, z2, np.less)
        array([[False,  True],
               [ True,  True]])
        >>> d_ref9.compare(z1, z2, np.greater_equal)
        array([[ True, False],
               [False, False]])
        >>> d_alpha4 = RankTransform(order='alphamod', alpha=4)
        >>> y1 = np.asarray([[[20, 10, 13], [43, 90, 51]],
        ...                  [[55, 20, 60], [56, 20, 60]]])
        ...
        >>> y2 = np.asarray([[[21, 10, 13], [42, 90, 50]],
        ...                  [[53, 20, 60], [53, 20, 60]]])
        ...
        >>> d_alpha4.compare(y1, y2, comp=np.less_equal)
        array([[ True, False],
               [ True, False]])

        """
        if self.order == 'linear':
            result = comp(xarr, yarr)
        elif self.order == 'product':
            result = np.all(comp(xarr, yarr), axis=-1)
        elif self.order == 'lexicographic':
            weights = np.asarray([256**np.arange(xarr.shape[2])[::-1]])
            lexic_x = np.sum(xarr[:, :, self.perm]*weights, axis=-1)
            lexic_y = np.sum(yarr[:, :, self.perm]*weights, axis=-1)
            result = comp(lexic_x, lexic_y)
        elif self.order == 'alphamod':
            xarr = xarr[:, :, self.perm]
            yarr = yarr[:, :, self.perm]
            xarr[:, :, 0] = xarr[:, :, 0] // self.alpha
            yarr[:, :, 0] = yarr[:, :, 0] // self.alpha
            weights = np.asarray([256**np.arange(xarr.shape[2])[::-1]])
            weights[0] = (256//self.alpha)**(xarr.shape[2] - 1)
            lexic_x = np.sum(xarr*weights, axis=-1)
            lexic_y = np.sum(yarr*weights, axis=-1)
            result = comp(lexic_x, lexic_y)
        elif self.order == 'bitmixing':
            mix_x = self.lut[tuple(xarr[:, :, self.perm].T)].T
            mix_y = self.lut[tuple(yarr[:, :, self.perm].T)].T
            result = comp(mix_x, mix_y)
        elif self.order == 'random':
            rand_x = self.lut[tuple(xarr.T)].T
            rand_y = self.lut[tuple(yarr.T)].T
            result = comp(rand_x, rand_y)
        elif self.order == 'refcolor':
            cref = np.asarray(self.cref).astype(np.int_)
            dist_x = np.linalg.norm(xarr - cref, axis=-1)
            dist_y = np.linalg.norm(yarr - cref, axis=-1)
            result = comp(dist_x, dist_y)
        return result


class Concatenation(object):
    """Class for concatenation of HEP descriptors."""


    def __init__(self, *descriptors):
        self.descriptors = descriptors


    def __call__(self, img):
        return np.concatenate([d(img) for d in self.descriptors])


    def __str__(self):
        return '+'.join([d.__str__() for d in self.descriptors])


class CompletedLocalBinaryCountSM(HEP):
    """Return the completed local binary count features.

    References
    ----------
    .. [1] Yang Zhao, De-Shuang Huang, and Wei Jia
           Completed Local Binary Count for Rotation Invariant Texture
           Classification
           https://doi.org/10.1109/TIP.2012.2204271
    """


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        if self.order in ('linear', 'lexicographic', 'alphamod',
                          'bitmixing', 'refcolor', 'random'):
            return [(p + 1)**2 for p in self.points]
        elif self.order == 'product':
            return [((p + 2)*(p + 1)//2)**2 for p in self.points]
        else:
            raise ValueError(
                '{} order is not supported for {} descriptor'.format(
                    self.order, self.__class__.__name__))


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the local concave and convex micro-structure patterns coding.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
        """
        central = img[radius: -radius, radius: -radius]
        codes_s = np.zeros(shape=central.shape[:2], dtype=np.int_)
        lt_s = np.zeros_like(codes_s)

        if self.order == 'product':
            ge_s = np.zeros_like(codes_s)

        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)

            lt_s += self.compare(neighbour, central, comp=np.less)

            if self.order == 'product':
                ge_s += self.compare(neighbour, central, comp=np.greater_equal)

        if self.order == 'product':
            dominated_s = lt_s
            non_comparable_s = points - dominated_s - ge_s
            codes_s = non_comparable_s + dominated_s*(2*points + 3 - dominated_s)//2
        else:
            codes_s = lt_s


        m_map = np.zeros(shape=central.shape, dtype=np.float_)
        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)
            m_map += np.abs(np.float_(neighbour) - central)
        m_map /= points
        if m_map.ndim > 2:
            mp_avg = np.array([*map(np.mean, tuple(m_map.T))])
            mp_avg = np.tile(mp_avg, m_map.shape[:2] + (1,))
        else:
            mp_avg = np.mean(m_map)
            mp_avg = np.tile(mp_avg, m_map.shape[:2])

        if self.order in ('bitmixing', 'random'):
            # Mean and median values need to be cast to integer in order to
            # be used as indices in the lut
            mp_avg = np.int_(mp_avg)

        codes_m = np.zeros(shape=central.shape[:2], dtype=np.int_)

        lt_m = np.zeros(shape=central.shape[:2], dtype=np.int_)

        if self.order == 'product':
            ge_m = np.zeros_like(codes_m)

        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)
            m_p = np.abs(np.float_(neighbour) - central)

            if self.order in ('bitmixing', 'random'):
            # Mean and median values need to be cast to integer in order to
            # be used as indices in the lut
                m_p = np.int_(m_p)

            lt_m += self.compare(m_p, mp_avg, comp=np.less)

            if self.order == 'product':
                ge_m += self.compare(m_p, mp_avg, comp=np.greater_equal)

        if self.order == 'product':
            dominated_m = lt_m
            non_comparable_m = points - dominated_m - ge_m
            codes_m = non_comparable_m + dominated_m*(2*points + 3 - dominated_m)//2
        else:
            codes_m = lt_m

        if self.order in ('linear', 'lexicographic', 'bitmixing',
                          'alphamod', 'refcolor', 'random'):
            ncodes = points + 1
        elif self.order == 'product':
            ncodes = (points + 2)*(points + 1)//2
        codes = codes_m + ncodes*codes_s

        return codes


class CompletedLocalBinaryCountSMC(HEP):
    """Return the completed local binary count features.

    References
    ----------
    .. [1] Yang Zhao, De-Shuang Huang, and Wei Jia
           Completed Local Binary Count for Rotation Invariant Texture
           Classification
           https://doi.org/10.1109/TIP.2012.2204271
    """


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        if self.order in ('linear', 'lexicographic', 'bitmixing',
                          'alphamod', 'refcolor', 'random'):
            return [2*(p + 1)**2 for p in self.points]
        elif self.order == 'product':
            return [3*((p + 2)*(p + 1)//2)**2 for p in self.points]
        else:
            raise ValueError(
                '{} order is not supported for {} descriptor'.format(
                    self.order, self.__class__.__name__))


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the local concave and convex micro-structure patterns coding.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
        """
        central = img[radius: -radius, radius: -radius]

        codes_s = np.zeros(shape=central.shape[:2], dtype=np.int_)

        lt_s = np.zeros_like(codes_s)

        if self.order == 'product':
            ge_s = np.zeros_like(codes_s)

        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)

            lt_s += self.compare(neighbour, central, comp=np.less)

            if self.order == 'product':
                ge_s += self.compare(neighbour, central, comp=np.greater_equal)

        if self.order == 'product':
            dominated_s = lt_s
            non_comparable_s = points - dominated_s - ge_s
            codes_s = non_comparable_s + dominated_s*(2*points + 3 - dominated_s)//2
        else:
            codes_s = lt_s

        # Magnitude descriptor
        m_map = np.zeros(shape=central.shape, dtype=np.float_)
        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)
            m_map += np.abs(np.float_(neighbour) - central)
        m_map /= points
        if m_map.ndim > 2:
            mp_avg = np.array([*map(np.mean, tuple(m_map.T))])
            mp_avg = np.tile(mp_avg, m_map.shape[:2] + (1,))
        else:
            mp_avg = np.mean(m_map)
            mp_avg = np.tile(mp_avg, m_map.shape[:2])

        if self.order in ('bitmixing', 'random'):
            # Mean and median values need to be cast to integer in order to
            # be used as indices in the lut
            mp_avg = np.int_(mp_avg)

        codes_m = np.zeros(shape=central.shape[:2], dtype=np.int_)

        lt_m = np.zeros(shape=central.shape[:2], dtype=np.int_)

        if self.order == 'product':
            ge_m = np.zeros_like(codes_m)

        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)
            m_p = np.abs(np.float_(neighbour) - central)

            if self.order in ('bitmixing', 'random'):
            # Mean and median values need to be cast to integer in order to
            # be used as indices in the lut
                m_p = np.int_(m_p)

            lt_m += self.compare(m_p, mp_avg, comp=np.less)

            if self.order == 'product':
                ge_m += self.compare(m_p, mp_avg, comp=np.greater_equal)

        if self.order == 'product':
            dominated_m = lt_m
            non_comparable_m = points - dominated_m - ge_m
            codes_m = non_comparable_m + dominated_m*(2*points + 3 - dominated_m)//2
        else:
            codes_m = lt_m

        # Center descriptor
        if central.ndim > 2:
            c_i = np.array([*map(np.mean, tuple(central.T))])
            c_i = np.tile(c_i, central.shape[:2] + (1,))
        else:
            c_i = np.mean(central)
            c_i = np.tile(c_i, central.shape[:2])
        if self.order in ('bitmixing', 'random'):
        # Mean and median values need to be cast to integer in order to
        # be used as indices in the lut
            c_i = np.int_(c_i)
            
        ge_c = self.compare(central, c_i, comp=np.greater_equal)
        if self.order == 'product':
            lt_c = self.compare(central, c_i, comp=np.less)
            # 3 possibilities: central < average (0)
            #                  central >= average (1)
            #                  central not comparable to average (2)
            codes_c = 2 * np.ones(shape=central.shape[:2], dtype=np.int_)
            codes_c[ge_c] = 1
            codes_c[lt_c] = 0
        else:
            codes_c = ge_c

        if self.order in ('linear', 'lexicographic', 'alphamod',
                          'bitmixing', 'refcolor', 'random'):
            n_m = points + 1
            n_c = 2
        elif self.order == 'product':
            n_m = (points + 2)*(points + 1)//2
            n_c = 3
        codes = codes_c + n_c*(codes_m + n_m*codes_s)
        return codes


class ImprovedCenterSymmetricLocalBinaryPattern(HEP):
    """Return the improved center-symmetric local bonary patterns features.

    References
    ----------
    .. [1] Xiaosheng Wu and Junding Sun
           An Effective Texture Spectrum Descriptor
           https://doi.org/10.1109/IAS.2009.126
    """


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        return [2**(p//2) for p in self.points]


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the improved center-symmetric local binary pattern coding.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
        """
        central = img[radius: -radius, radius: -radius]
        codes = np.zeros(shape=central.shape[:2], dtype=np.int_)

        for exponent, index in enumerate(range(points//2)):

            start = utils.subimage(img, index, radius)
            end = utils.subimage(img, index + points//2, radius)

            less1 = self.compare(start, central, comp=np.less)
            less2 = self.compare(central, end, comp=np.less)
            greq1 = self.compare(start, central, comp=np.greater_equal)
            greq2 = self.compare(central, end, comp=np.greater_equal)

            greq = np.logical_and(greq1, greq2)
            less = np.logical_and(less1, less2)
            codes += np.logical_or(greq, less)*2**exponent
        return codes


class LocalConcaVeMicroStructurePatterns(HEP):
    """Return the local concave micro-structure patterns.

    References
    ----------
    .. [1] Y. El merabet, Y. Ruichek
           Local Concave-and-Convex Micro-Structure Patterns for texture
           classification
           https://doi.org/10.1016/j.patcog.2017.11.005
    """


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        return [2**(p + 2) for p in self.points]


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the local concave micro-structure patterns coding.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
        """
        central = img[radius: -radius, radius: -radius]
        codes = np.zeros(shape=central.shape[:2], dtype=np.int_)

        for pixel in range(points):

            first = utils.subimage(img, pixel, radius)
            last = utils.subimage(img, (pixel + 2)% points, radius)

            ge_first = self.compare(first, central, np.greater_equal)
            ge_last = self.compare(last, central, np.greater_equal)

            codes += np.logical_and(ge_first, ge_last)*2**pixel

        return codes


class LocalConveXMicroStructurePatterns(HEP):
    """Return the local convex micro-structure patterns.

    References
    ----------
    .. [1] Y. El merabet, Y. Ruichek
           Local Concave-and-Convex Micro-Structure Patterns for texture
           classification
           https://doi.org/10.1016/j.patcog.2017.11.005
    """


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        return [2**(p + 2) for p in self.points]


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the local convex micro-structure patterns coding.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
        """
        central = img[radius: -radius, radius: -radius]
        codes = np.zeros(shape=central.shape[:2], dtype=np.int_)

        for pixel in range(points):

            first = utils.subimage(img, pixel, radius)
            last = utils.subimage(img, (pixel + 2)% points, radius)

            le_first = self.compare(first, central, np.less_equal)
            le_last = self.compare(last, central, np.less_equal)

            codes += np.logical_and(le_first, le_last)*2**pixel

        return codes


class LocalConcaveConvexMicroStructurePatterns(HEP):
    """Return the local concave and convex micro-structure patterns.

    References
    ----------
    .. [1] Y. El merabet, Y. Ruichek
           Local Concave-and-Convex Micro-Structure Patterns for texture
           classification
           https://doi.org/10.1016/j.patcog.2017.11.005
    """


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        return [2*2**(p + 2) for p in self.points]


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the local concave and convex micro-structure patterns coding.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
        """
        step = 2

        central = img[radius: -radius, radius: -radius]

        if img.ndim == 3:
            global_mean = np.asarray([*map(np.mean, img.T)])
            global_median = np.asarray([*map(np.median, img.T)])
        elif img.ndim == 2:
            global_mean = np.mean(img)
            global_median = np.median(img)
        else:
            raise ValueError("Image has to be 2D or 3D array")

        global_mean = np.zeros(shape=central.shape) + global_mean
        global_median = np.zeros(shape=central.shape) + global_median

        lst = [utils.subimage(img, pixel, radius) for pixel in range(points)]
        lst.append(central)
        neighbours = np.stack(lst, axis=-1)
        local_mean = np.mean(neighbours, axis=-1)
        local_median = np.median(neighbours, axis=-1)

        if self.order in ('bitmixing', 'random'):
            # Mean and median values need to be cast to integer in order to
            # be used as indices in the lut
            local_mean = np.int_(local_mean)
            local_median = np.int_(local_median)
            global_mean = np.int_(global_mean)
            global_median = np.int_(global_median)

        codes_concave = np.zeros(shape=central.shape[:2], dtype=np.int_)
        codes_convex = np.zeros(shape=central.shape[:2], dtype=np.int_)

        for pixel in range(points)[::-1]:

            first = utils.subimage(img, pixel, radius)
            last = utils.subimage(img, (pixel - step) % points, radius)

            ge_first = self.compare(first, central, np.greater_equal)
            ge_last = self.compare(last, central, np.greater_equal)
            le_first = self.compare(first, central, np.less_equal)
            le_last = self.compare(last, central, np.less_equal)

#            ge_first = d1.compare(first, central, np.greater_equal)
#            ge_last = d1.compare(last, central, np.greater_equal)
#            le_first = d1.compare(first, central, np.less_equal)
#            le_last = d1.compare(last, central, np.less_equal)

#            ge_first = np.greater_equal(first, central)
#            ge_last = np.greater_equal(last, central)
#            le_first = np.less_equal(first, central)
#            le_last = np.less_equal(last, central)

            codes_concave += np.logical_and(ge_first, ge_last)*2**pixel
            codes_convex += np.logical_and(le_first, le_last)*2**pixel

        ge_first = self.compare(local_mean, central, np.greater_equal)
        ge_last = self.compare(local_median, central, np.greater_equal)
        le_first = self.compare(local_mean, central, np.less_equal)
        le_last = self.compare(local_median, central, np.less_equal)

#        ge_first = d1.compare(local_mean, central, np.greater_equal)
#        ge_last = d1.compare(local_median, central, np.greater_equal)
#        le_first = d1.compare(local_mean, central, np.less_equal)
#        le_last = d1.compare(local_median, central, np.less_equal)

        codes_concave += np.logical_and(ge_first, ge_last)*2**points
        codes_convex += np.logical_and(le_first, le_last)*2**points

        ge_first = self.compare(global_mean, central, np.greater_equal)
        ge_last = self.compare(global_median, central, np.greater_equal)
        le_first = self.compare(global_mean, central, np.less_equal)
        le_last = self.compare(global_median, central, np.less_equal)

#        ge_first = np.greater_equal(global_mean, central)
#        ge_last = np.greater_equal(global_median, central)
#        le_first = np.less_equal(global_mean, central)
#        le_last = np.less_equal(global_median, central)

        codes_concave += np.logical_and(ge_first, ge_last)*2**(points + 1)
        codes_convex += np.logical_and(le_first, le_last)*2**(points + 1)

        codes = np.hstack([codes_concave, codes_convex + 2**(points + 2)])

        return codes


class LocalDirectionalRankCoding(HEP):
    """Return the local directional rank coding.

    References
    ----------
    .. [1] Farida Ouslimani, Achour Ouslimani and Zohra Ameur
           Directional Rank Coding For Multi-scale Texture Classification
           https://hal.archives-ouvertes.fr/hal-01368416
    """


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        if self.order in ('linear', 'lexicographic', 'alphamod',
                          'bitmixing', 'refcolor', 'random'):
            return [3**(p//2) for p in self.points]
        elif self.order == 'product':
            return [4**(p//2) for p in self.points]
        else:
            msg = '''{} order is not supported for {} descriptor'''.format(
                self.order, self.__class__.__name__)
            raise ValueError(msg)


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the local directional rank coding.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.

        """
        central = img[radius: -radius, radius: -radius]
        codes = np.zeros(shape=central.shape[:2], dtype=np.int_)

        for exponent, index in enumerate(range(points//2)):

            rank = np.zeros_like(codes)

            less = np.zeros_like(codes)
            if self.order == 'product':
                greq = np.zeros_like(codes)

            for pixel in [index, index + points//2]:

                neighbour = utils.subimage(img, pixel, radius)

                less += self.compare(neighbour, central, np.less)

                if self.order == 'product':
                    greq += self.compare(neighbour, central, np.greater_equal)

            if self.order == 'product':
                rank = np.where(less + greq < 2, 3, less)
                codes += rank*4**exponent
            else:
                rank = less
                codes += rank*3**exponent

        return codes


class RankTransform(HEP):
    """Return the rank transform.

    References
    ----------
    .. [1] R. Zabih and J. Woodfill
           Non-parametric local transforms for computing visual correspondence
           https://doi.org/10.1007/BFb0028345
    .. [2] A. Fernández, D. Lima, F. Bianconi and F. Smeraldi
           Compact color texture descriptor based on rank transform and
           product ordering in the RGB color space
           https://doi.org/10.1109/ICCVW.2017.126
           
    Examples
    --------
    >>> rt_lin = RankTransform(order='linear', radius=[1, 2, 3])
    >>> rt_lin.dims
    [9, 17, 25]
    >>> rt_prod = RankTransform(order='product', radius=[1, 2, 3])
    >>> rt_prod.dims
    [45, 153, 325]
    >>> rt_lex = RankTransform(order='lexicographic', radius=[1, 2, 3])
    
    """
    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        if self.order in ('linear', 'lexicographic', 'alphamod',
                          'bitmixing', 'refcolor', 'random'):
            return [p + 1 for p in self.points]
        elif self.order == 'product':
            return [(p + 2)*(p + 1)//2 for p in self.points]
        else:
            self.print_order_not_supported()


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the rank transform.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
            
        """
        central = img[radius: -radius, radius: -radius]
        less = np.zeros(shape=central.shape[:2], dtype=np.int_)

        if self.order == 'product':
            greq = np.zeros_like(less)

        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)
            less += self.compare(neighbour, central, comp=np.less)

            if self.order == 'product':
                greq += self.compare(neighbour, central, comp=np.greater_equal)

        if self.order == 'product':
            dominated = less
            non_comparable = points - dominated - greq
            codes = non_comparable + dominated*(2*points + 3 - dominated)//2
        else:
            codes = less

        return codes


if __name__ == '__main__':   
    # Run tests
    import doctest
    doctest.testmod()