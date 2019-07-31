#! /usr/bin/env python37

"""Methods for characterizing texture images through
Histogram of Equivalent Patterns (HEP).
"""


import numpy as np

import utils


####################
# HELPER FUNCTIONS #
####################


def square_frame_size(radius):
    """Compute the number of pixels of the external frame of square
    neighbourhoods of the given radii.

    Parameters
    ----------
    radius : list of int
        Radii of the local neighborhoods.

    Returns
    -------
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
    _canonical_orders = ('linear', 
                         'lexicographic', 
                         'alphamod', 
                         'bitmixing', 
                         'refcolor', 
                         'random')
    _product_orders = ('product',)


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
        radius : list of int, optional (default is [1])
            Radii of the local neighborhoods.
        bands : str, optional (default is 'RGB')
            Color band priorities.
        cref : list of int, optional (default is [0, 0, 0])
            RGB coordinates of the reference color.
        seed : int, optional (default is 0)
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


    def raise_order_not_supported(self):
        raise ValueError(
                f'{self.order} order not supported for {self.abbrev()}')

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
        descriptor : hep.HEP
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
        descr : hep.HEP
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
        xarr, yarr : array
            Images to be compared.
        comp : NumPy function, optional (default `np.less`)
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
        >>> d_lin.compare(i1, i2, np.less_equal)
        array([[ True,  True, False],
               [ True, False, False]])
        >>> d_lin.compare(i1, i2, np.greater_equal)
        array([[ True, False,  True],
               [False,  True,  True]])
        >>> a1 = np.array([[[0, 1, 2], 
        ...                 [0, 1, 2]],
        ...                [[3, 4, 5], 
        ...                 [3, 4, 5]]])
        ...
        >>> a2 = np.asarray([[[2, 2, 2], 
        ...                   [3, 3, 3]], 
        ...                  [[3, 3, 3], 
        ...                   [2, 2, 2]]])
        ...
        >>> d_prod = RankTransform(order='product')
        >>> d_prod.compare(a1, a2, np.greater)
        array([[False, False],
               [False,  True]])
        >>> d_prod.compare(a1, a2)
        array([[False,  True],
               [False, False]])
        >>> b1 = np.array([[[0, 1, 2], 
        ...                 [0, 1, 2], 
        ...                 [1, 2, 1]],
        ...                [[3, 4, 5], 
        ...                 [6, 7, 8],
        ...                 [2, 3, 3]]])
        ...
        >>> b2 = np.array([[[2, 0, 1], 
        ...                 [0, 1, 2], 
        ...                 [0, 1, 2]],
        ...                [[4, 5, 6], 
        ...                 [8, 8, 8],
        ...                 [2, 3, 4]]])
        ...
        >>> d_lexrgb = RankTransform(order='lexicographic', bands='RBG')
        >>> d_lexrgb.compare(b1, b2)
        array([[ True, False, False],
               [ True,  True,  True]])
        >>> d_lexbgr = RankTransform(order='lexicographic', bands='BGR')
        >>> d_lexbgr.compare(b1, b2, np.greater)
        array([[ True, False, False],
               [False, False, False]])
        >>> c1 = np.array([[[0b00000001, 0b00000010, 0b00000100]],
        ...                [[0b11110000, 0b00111100, 0b00001111]],
        ...                [[0b10000000, 0b11000000, 0b11100000]]])
        ...
        >>> c2 = np.array([[[0b00000001, 0b00000011, 0b00000111]],
        ...                [[0b11110000, 0b00111100, 0b00001111]],
        ...                [[0b10000000, 0b10000000, 0b10000000]]])
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
        >>> z1 = np.array([[[0, 0, 1], 
        ...                 [0, 3, 4]],
        ...                [[3, 4, 0], 
        ...                 [1, 1, 1]]])
        ...
        >>> z2 = np.array([[[0, 0, 1], 
        ...                 [5, 0, 0]],
        ...                [[1, 1, 1], 
        ...                 [0, 3, 0]]])
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
        >>> x1 = np.array([[[20, 10, 13], 
        ...                 [43, 90, 51]],
        ...                [[55, 20, 60], 
        ...                 [56, 20, 60]]])
        ...
        >>> x2 = np.array([[[21, 10, 13], 
        ...                 [42, 90, 50]],
        ...                [[53, 20, 60], 
        ...                 [53, 20, 60]]])
        ...
        >>> d_alpha4.compare(x1, x2, comp=np.less_equal)
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


class CompletedLocalBinaryCountSM(HEP):
    """Return the completed local binary count features (sign and magnitude).

    Examples
    --------
    >>> sm = CompletedLocalBinaryCountSM(order='linear', radius=[1, 2, 3])
    >>> sm.dims
    [81, 289, 625]
    >>> gray = np.array([[ 25, 33,  80,  94],
    ...                  [141, 25, 175, 120],
    ...                  [ 15, 17,  12, 203]], dtype=np.uint8)
    ...
    >>> sm.codemap(gray, 1, 8)
    array([[33, 65]])
    >>> smprod = CompletedLocalBinaryCountSM(order='product', radius=[1, 2, 3])
    >>> smprod.dims
    [2025, 23409, 105625]
    >>> rgb = np.array([[[134,  98, 245],
    ...                  [242, 123, 232],
    ...                  [246,  98, 117],
    ...                  [228, 164,  54]],
    ...                 [[126,  75, 165],
    ...                  [227, 241, 145],
    ...                  [160,  38,  32],
    ...                  [ 47,  10, 185]],
    ...                 [[142, 246,  83],
    ...                  [ 69, 153, 234],
    ...                  [ 10, 118,  24],
    ...                  [114,  94,  30]]], dtype=np.uint8)
    ...
    >>> smprod.codemap(rgb, 1, 8)
    array([[1049,  203]])
    
    References
    ----------
    .. [1] Yang Zhao, De-Shuang Huang, and Wei Jia
           Completed Local Binary Count for Rotation Invariant Texture
           Classification
           https://doi.org/10.1109/TIP.2012.2204271
    
    """

    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        if self.order in self._canonical_orders:
            return [(p + 1)**2 for p in self.points]
        elif self.order in self._product_orders:
            return [((p + 2)*(p + 1)//2)**2 for p in self.points]
        else:
            self.raise_order_not_supported()


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

        # Sign descriptor
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
            non_comp_s = points - dominated_s - ge_s
            codes_s = non_comp_s + dominated_s*(2*points + 3 - dominated_s)//2
        else:
            codes_s = lt_s

        # Magnitude descriptor
        m_map = np.zeros(shape=central.shape, dtype=np.float_)
        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)
            m_map += np.abs(np.float_(neighbour) - central)
        m_map /= points
       
        if m_map.ndim == 3:
            mp_avg = np.array([*map(np.mean, tuple(m_map.T))])
            mp_avg = np.tile(mp_avg, m_map.shape[:2] + (1,))
        elif m_map.ndim == 2:
            mp_avg = np.full(central.shape, np.mean(m_map))
        else:
            raise ValueError("Image has to be 2D or 3D array")

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
            non_comp_m = points - dominated_m - ge_m
            codes_m = non_comp_m + dominated_m*(2*points + 3 - dominated_m)//2
        else:
            codes_m = lt_m

        # Joint sign and magnitude descriptor
        if self.order in self._canonical_orders:
            num_codes_m = points + 1
        elif self.order in self._product_orders:
            num_codes_m = (points + 2)*(points + 1)//2
        codes = num_codes_m*codes_s + codes_m

        return codes


class CompletedLocalBinaryCountSMC(CompletedLocalBinaryCountSM):
    """Return the completed local binary count (sign, magnitude and center).
    
    To avoid unnecessary repetition of code this class inherits most 
    of its behaviour from `CompletedLocalBinaryCountSM`.
    
    Examples
    --------
    >>> smc = CompletedLocalBinaryCountSMC(order='linear', radius=[1, 2, 3])
    >>> smc.dims
    [162, 578, 1250]
    >>> gray = np.array([[ 25, 33,  80,  94],
    ...                  [141, 25, 175, 120],
    ...                  [ 15, 17,  12, 203]], dtype=np.uint8)
    ...
    >>> smc.codemap(gray, 1, 8)
    array([[ 66, 131]])
    >>> smcp = CompletedLocalBinaryCountSMC(order='product', radius=[1, 2, 3])
    >>> smcp.dims
    [6075, 70227, 316875]
    >>> rgb = np.array([[[134,  98, 245],
    ...                  [242, 123, 232],
    ...                  [246,  98, 117],
    ...                  [228, 164,  54]],
    ...                 [[126,  75, 165],
    ...                  [227, 241, 145],
    ...                  [160,  38,  32],
    ...                  [ 47,  10, 185]],
    ...                 [[142, 246,  83],
    ...                  [ 69, 153, 234],
    ...                  [ 10, 118,  24],
    ...                  [114,  94,  30]]], dtype=np.uint8)
    ...
    >>> smcp.codemap(rgb, 1, 8)
    array([[3148,  609]])

    References
    ----------
    .. [1] Yang Zhao, De-Shuang Huang, and Wei Jia
           Completed Local Binary Count for Rotation Invariant Texture
           Classification
           https://doi.org/10.1109/TIP.2012.2204271

    """
    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        if self.order in self._canonical_orders:
            return [2*(p + 1)**2 for p in self.points]
        elif self.order in self._product_orders:
            return [3*((p + 2)*(p + 1)//2)**2 for p in self.points]
        else:
            self.raise_order_not_supported()


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

        # Center descriptor
        if central.ndim == 3:
            c_i = np.array([*map(np.mean, tuple(central.T))])
            c_i = np.tile(c_i, central.shape[:2] + (1,))
        elif central.ndim == 2:
            c_i = np.full(central.shape, np.mean(central))
        else:
            raise ValueError("Image has to be 2D or 3D array")

        if self.order in ('bitmixing', 'random'):
        # Mean and median values need to be cast to integer in order to
        # be used as indices in the lut
            c_i = np.int_(c_i)
            
        ge_c = self.compare(central, c_i, comp=np.greater_equal)
        if self.order == 'product':
            lt_c = self.compare(central, c_i, comp=np.less)
            # 3 possibilities: central < average -> 0
            #                  central >= average -> 1
            #                  central not comparable to average -> 2
            codes_c = np.full(central.shape[:2], fill_value=2, dtype=np.int_)
            codes_c[ge_c] = 1
            codes_c[lt_c] = 0
        else:
            codes_c = ge_c

        codes_sm = super().codemap(img, radius, points)

        if self.order in self._canonical_orders:
            num_codes_c = 2
        elif self.order in self._product_orders:
            num_codes_c = 3

        codes = codes_c + num_codes_c*codes_sm
        
        return codes


class ImprovedCenterSymmetricLocalBinaryPattern(HEP):
    """Return the improved center-symmetric local binary patterns features.

    Examples
    --------
    >>> params_lin = dict(order='linear', radius=[1, 2, 3])
    >>> icslin = ImprovedCenterSymmetricLocalBinaryPattern(**params_lin)
    >>> icslin.dims
    [16, 256, 4096]
    >>> gray = np.array([[ 25, 33,  80,  94, 114],
    ...                  [141, 25, 175, 120,  24],
    ...                  [ 15, 17,  12, 203, 120]], dtype=np.uint8)
    ...
    >>> icslin.codemap(gray, 1, 8)
    array([[ 6,  8, 13]])
    >>> args_lex_rgb = dict(order='lexicographic', radius=[1], bands='RGB')
    >>> icslexrgb = ImprovedCenterSymmetricLocalBinaryPattern(**args_lex_rgb)
    >>> rgb = np.array([[[134,  98, 245],
    ...                  [242, 123, 232],
    ...                  [246,  98, 117],
    ...                  [228, 164,  54]],
    ...                 [[126,  75, 165],
    ...                  [227, 241, 145],
    ...                  [160,  38,  32],
    ...                  [ 47,  10, 185]],
    ...                 [[142, 246,  83],
    ...                  [ 69, 153, 234],
    ...                  [ 10, 118,  24],
    ...                  [114,  34,  30]]], dtype=np.uint8)
    >>> icslexrgb.codemap(rgb, 1, 8)
    array([[ 6, 15]])
    >>> args_lex_bgr = dict(order='lexicographic', radius=[1], bands='BGR')
    >>> icslexbgr = ImprovedCenterSymmetricLocalBinaryPattern(**args_lex_bgr)
    >>> icslexbgr.codemap(rgb, 1, 8)
    array([[ 9, 12]])
    >>> args_prod = dict(order='product', radius=[1, 2, 3])
    >>> icsprod = ImprovedCenterSymmetricLocalBinaryPattern(**args_prod)
    >>> icsprod.dims
    [16, 256, 4096]
    >>> icsprod.codemap(rgb, 1, 8)
    array([[0, 8]])

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

            ge_1 = self.compare(start, central, comp=np.greater_equal)
            ge_2 = self.compare(central, end, comp=np.greater_equal)
            lt_1 = self.compare(start, central, comp=np.less)
            lt_2 = self.compare(central, end, comp=np.less)

            ge = np.logical_and(ge_1, ge_2)
            lt = np.logical_and(lt_1, lt_2)
            codes += np.logical_or(ge, lt)*2**exponent
        return codes


class MicroStructurePatterns(HEP):
    """Superclass for micro structure pattern models
    
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


    def codemap(self, img, radius, points, comp):
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
        comp : NumPy function
            Comparison function (`np.greater_equal` or `np.less_equal`).

        Returns
        -------
        codes : array
            Map of feature values.
            
        """
        ndims = img.ndim
        if ndims not in (2, 3):
            raise ValueError("Image has to be 2D or 3D array")

        step = 2
        central = img[radius: -radius, radius: -radius]
        codes = np.zeros(shape=central.shape[:2], dtype=np.int_)

        crops = [central]

        for pixel in range(points):

            first = utils.subimage(img, pixel, radius)
            last = utils.subimage(img, (pixel + step) % points, radius)

            crops.append(first)

            comp_first = self.compare(first, central, comp)
            comp_last = self.compare(last, central, comp)

            codes += np.logical_and(comp_first, comp_last)*2**pixel

        crops = np.stack(crops, axis=-1)
       
        # Contribution of the local statistics         
        if ndims > 2:
            avg_patch = np.mean(crops, axis=-1)
            med_patch = np.median(crops, axis=-1)
        else:
            avg_patch = np.mean(crops, axis=-1)
            med_patch = np.median(crops, axis=-1)

        if self.order in ('bitmixing', 'random'):
            # Mean and median values need to be cast to integer in order to
            # be used as indices in the lut
            avg_patch = np.int_(avg_patch)
            med_patch = np.int_(med_patch)

        comp_avg_patch = self.compare(avg_patch, central, comp)
        comp_med_patch = self.compare(med_patch, central, comp)
        
        codes += np.logical_and(comp_avg_patch, comp_med_patch)*2**points

        # Contribution of the global statistics
        if ndims > 2:
            avg_img = np.array([*map(np.mean, tuple(img.T))])
            avg_img = np.tile(avg_img, central.shape[:2] + (1,))
            med_img = np.array([*map(np.median, tuple(img.T))])
            med_img = np.tile(med_img, central.shape[:2] + (1,))
        else:
            avg_img = np.full(central.shape, np.mean(img))
            med_img = np.full(central.shape, np.median(img))
        
        if self.order in ('bitmixing', 'random'):
            # Mean and median values need to be cast to integer in order to
            # be used as indices in the lut
            avg_img = np.int_(avg_img)
            med_img = np.int_(med_img)

        comp_avg_img = self.compare(avg_img, central, comp)
        comp_med_img = self.compare(med_img, central, comp)
        
        codes += np.logical_and(comp_avg_img, comp_med_img)*2**(points + 1)

        return codes


class LocalConcaVeMicroStructurePatterns(MicroStructurePatterns):
    """Return the local concave micro-structure patterns.

    Examples
    --------
    >>> params_lin = dict(order='linear', radius=[1, 2, 3])
    >>> lcvlin = LocalConcaVeMicroStructurePatterns(**params_lin)
    >>> lcvlin.dims
    [1024, 262144, 67108864]
    >>> gray = np.array([[ 25,  33,  80,  94, 114],
    ...                  [141,  25, 175, 120,  24],
    ...                  [ 15,  17,  12, 203, 120],
    ...                  [ 36, 102,  94, 186, 203]], dtype=np.uint8)
    ...
    >>> lcvlin.codemap(gray, 1, 8)
    array([[ 880,    0,    1],
           [ 938, 1023,    0]])
    >>> args_lex_gbr = dict(order='lexicographic', radius=[1], bands='GBR')
    >>> lcvlexgbr = LocalConcaVeMicroStructurePatterns(**args_lex_gbr)
    >>> rgb = np.array([[[6, 5, 9],
    ...                  [9, 6, 9],
    ...                  [9, 4, 5],
    ...                  [8, 7, 2]],
    ...                 [[6, 3, 8],
    ...                  [9, 9, 7],
    ...                  [8, 1, 1],
    ...                  [2, 0, 9]],
    ...                 [[7, 9, 4],
    ...                  [2, 6, 9],
    ...                  [0, 5, 1],
    ...                  [5, 1, 1]], 
    ...                 [[4, 5, 6],
    ...                  [3, 0, 0],
    ...                  [6, 3, 4],
    ...                  [1, 2, 7]]], dtype=np.uint8)
    ...
    >>> lcvlexgbr.codemap(rgb, 1, 8)
    array([[  0, 993],
           [ 64,   0]])
    >>> args_prod = dict(order='product', radius=[1, 2, 3])
    >>> lcvprod = LocalConcaVeMicroStructurePatterns(**args_prod)
    >>> lcvprod.dims
    [1024, 262144, 67108864]
    >>> lcvprod.codemap(rgb, 1, 8)
    array([[ 0, 96],
           [ 0,  0]])

    """
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
        return super().codemap(img, radius, points, comp=np.greater_equal)


class LocalConveXMicroStructurePatterns(MicroStructurePatterns):
    """Return the local convex micro-structure patterns.

    Examples
    --------
    >>> params_lin = dict(order='linear', radius=[1, 2, 3])
    >>> lcxlin = LocalConveXMicroStructurePatterns(**params_lin)
    >>> lcxlin.dims
    [1024, 262144, 67108864]
    >>> gray = np.array([[ 25,  33,  80,  94, 114],
    ...                  [141,  25, 175, 120,  24],
    ...                  [ 15,  17,  12, 203, 120],
    ...                  [ 36, 102,  94, 186, 203]], dtype=np.uint8)
    ...
    >>> lcxlin.codemap(gray, 1, 8)
    array([[ 130, 1013,  954],
           [   0,    0, 1023]])
    >>> args_lex_gbr = dict(order='lexicographic', radius=[1], bands='GBR')
    >>> lcxlexgbr = LocalConveXMicroStructurePatterns(**args_lex_gbr)
    >>> rgb = np.array([[[6, 5, 9],
    ...                  [9, 6, 9],
    ...                  [9, 4, 5],
    ...                  [8, 7, 2]],
    ...                 [[6, 3, 8],
    ...                  [9, 9, 7],
    ...                  [8, 1, 1],
    ...                  [2, 0, 9]],
    ...                 [[7, 9, 4],
    ...                  [2, 6, 9],
    ...                  [0, 5, 1],
    ...                  [5, 1, 1]], 
    ...                 [[4, 5, 6],
    ...                  [3, 0, 0],
    ...                  [6, 3, 4],
    ...                  [1, 2, 7]]], dtype=np.uint8)
    ...
    >>> lcxlexgbr.codemap(rgb, 1, 8)
    array([[1023,    0],
           [ 942,  798]])
    >>> args_prod = dict(order='product', radius=[1, 2, 3])
    >>> lcxprod = LocalConveXMicroStructurePatterns(**args_prod)
    >>> lcxprod.dims
    [1024, 262144, 67108864]
    >>> lcxprod.codemap(rgb, 1, 8)
    array([[778,   0],
           [  0,   0]])

    """
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
        return super().codemap(img, radius, points, comp=np.less_equal)


class LocalConcaveConvexMicroStructurePatterns(MicroStructurePatterns):
    """Return the local concave and convex micro-structure patterns.
    
    Attributes
    ----------
    components : list of HEP
        Descriptors to be concatenated.

    Examples
    --------
    >>> params_prod = dict(order='product', radius=[1])
    >>> lcc_prod = LocalConcaveConvexMicroStructurePatterns(**params_prod)
    >>> lcc_prod.dims
    [2048]
    >>> lcv_prod = LocalConcaVeMicroStructurePatterns(**params_prod)
    >>> lcx_prod = LocalConveXMicroStructurePatterns(**params_prod)
    >>> img = np.random.randint(0, high=256, size=(60, 80, 3), dtype=np.uint8)
    >>> hv_prod = lcv_prod(img)
    >>> hx_prod = lcx_prod(img)
    >>> hc_prod = lcc_prod(img)
    >>> np.array_equal(hc_prod, np.concatenate([hv_prod, hx_prod]))
    True
    >>> params_mix = dict(order='bitmixing', radius=[1], bands='BGR')
    >>> lcc_mix = LocalConcaveConvexMicroStructurePatterns(**params_mix)
    >>> lcv_mix = LocalConcaVeMicroStructurePatterns(**params_mix)
    >>> lcx_mix = LocalConveXMicroStructurePatterns(**params_mix)
    >>> hv_mix = lcv_mix(img)
    >>> hx_mix = lcx_mix(img)
    >>> hc_mix = lcc_mix(img)
    >>> np.array_equal(hc_mix, np.concatenate([hv_mix, hx_mix]))
    True

    """
    def __init__(self, **kwargs):
        """Initializer of the class
        
        Parameters
        ----------
        kwargs : dictionary
            Parameters the descriptor depends on, such as `order`, 
            `radius`, `bands`, etc.
            
        """
        super().__init__(**kwargs)
        self.components = [LocalConcaVeMicroStructurePatterns, 
                           LocalConveXMicroStructurePatterns]

        
    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        return [2*2**(p + 2) for p in self.points]


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
        params = {k: v for k, v in self.__dict__.items()}
        histograms = []
        for component in self.components:
            descriptor = component(**params)
            histograms.append(descriptor(img))
        return np.concatenate(histograms)


class LocalDirectionalRankCoding(HEP):
    """Return the local directional rank coding.

    Examples
    --------
    >>> dlin = LocalDirectionalRankCoding(order='linear', radius=[1, 2, 3])
    >>> dlin.dims
    [81, 6561, 531441]
    >>> gray = np.array([[25, 33, 10],
    ...                  [53, 25, 75],
    ...                  [15, 17, 12]])
    ...
    >>> dlin.codemap(gray, radius=1, points=8)
    array([[42]])
    >>> dp = LocalDirectionalRankCoding(order='product', radius=[1, 2, 3])
    >>> dp.dims
    [256, 65536, 16777216]
    >>> rgb = np.array([[[102, 220, 225],
    ...                  [ 95, 179,  61],
    ...                  [234, 203,  92]],
    ...                 [[  3,  98, 243],
    ...                  [ 14, 149, 245],
    ...                  [ 46, 186, 250]],
    ...                 [[ 99, 187,  71],
    ...                  [212, 153, 199],
    ...                  [188, 174,  65]]])
    ...
    >>> dp.codemap(rgb, radius=1, points=8)
    array([[253]])
        
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
            self.raise_order_not_supported()


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
            lt = np.zeros_like(codes)

            if self.order == 'product':
                ge = np.zeros_like(codes)

            for pixel in [index, index + points//2]:
                neighbour = utils.subimage(img, pixel, radius)
                lt += self.compare(neighbour, central, np.less)

                if self.order == 'product':
                    ge += self.compare(neighbour, central, np.greater_equal)

            if self.order == 'product':
                rank = np.where(lt + ge < 2, 3, lt)
                codes += rank*4**exponent
            else:
                rank = lt
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
    >>> gray = np.array([[25, 33, 80],
    ...                  [53, 25, 75],
    ...                  [15, 17, 12]])
    ...
    >>> rt_lin.codemap(gray, 1, 8)
    array([[3]])
    >>> rt_prod = RankTransform(order='product', radius=[1, 2, 3])
    >>> rt_prod.dims
    [45, 153, 325]
    
    """
    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        if self.order in self._canonical_orders:
            return [p + 1 for p in self.points]
        elif self.order in self._product_orders:
            return [(p + 2)*(p + 1)//2 for p in self.points]
        else:
            self.raise_order_not_supported()


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
        lt = np.zeros(shape=central.shape[:2], dtype=np.int_)

        if self.order == 'product':
            ge = np.zeros_like(lt)

        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)
            lt += self.compare(neighbour, central, comp=np.less)

            if self.order == 'product':
                ge += self.compare(neighbour, central, comp=np.greater_equal)

        if self.order == 'product':
            dominated = lt
            non_comparable = points - dominated - ge
            codes = non_comparable + dominated*(2*points + 3 - dominated)//2
        else:
            codes = lt

        return codes


if __name__ == '__main__':   
    # Run tests
    import doctest
    doctest.testmod()