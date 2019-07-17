#! /usr/bin/env python3
#pylint: disable=W0223

"""Datasets classes for texture classification."""

import os
import imghdr
import numpy as np


####################
# Helper functions #
####################


def find_images(dirpath):
    """Recursively detect image files contained in a folder.

    Parameters
    ----------
    dirpath : string
        Full path name of the folder that contains the images.

    Returns
    -------
    imgfiles : list
        Full path names of all the image files in `dirpath` (and its
        subfolders).

    Notes
    -----
    The complete list of image types being detected can be found at:
    https://docs.python.org/3/library/imghdr.html
    
    """
    imgfiles = [os.path.join(root, filename)
                for root, dirs, files in os.walk(dirpath)
                for filename in files
                if imghdr.what(os.path.join(root, filename))]
    return imgfiles


###########
# Classes #
###########

    
class TextureDataset(object):
    """Superclass for texture datasets."""


    def __init__(self, dirpath):
        """Initializer of a TextureDataset instance.

        Parameters
        ----------
        dirpath : string
            Full path to the folder that contains the dataset. The name 
            of the last path component (i.e. the name of the folder that 
            contains the images) must match the class name.
            
        """
        self.images = find_images(dirpath)
        self.classes = {text: num for num, text in enumerate(sorted(
            set(self.get_class(img) for img in self.images)))}
        self.labels = np.asarray([self.get_label(img) for img in self.images])


    def get_label(self, img):
        """Return the numeric label associated with the class of an image."""
        return self.classes[self.get_class(img)]


    def get_class(self, img):
        raise NotImplementedError("Subclasses should implement this!")


    def get_rotation(self, img):
        raise NotImplementedError("Subclasses should implement this!")


    def get_scale(self, img):
        raise NotImplementedError("Subclasses should implement this!")


    def get_viewpoint(self, img):
        raise NotImplementedError("Subclasses should implement this!")


    def __str__(self):
        return self.__class__.__name__


class CBT(TextureDataset):
    """Class for Coloured Brodatz Textures dataset.

    Notes
    -----
    CBT is a colourised version of Brodatz's album. There are 112 
    classes (labeled from `D001` to `D112`), one image sample for 
    each class, which has been subdivided into 16 non-overlapping 
    sub-images (numbered from 1 to 16), resulting in a total of 
    1792 samples. It should be noted that a few samples of classes 
    D043 and D044 are flat. Classes D039, D040, D041, D042, D043, 
    D044, D045, D059, D062, D069, D070 and D097 are not stationary 
    textures.

    Image format : .tif (RGB)
    Sample size : 160x160 px

    >>> import config
    >>> cbt = CBT(os.path.join(config.imgs, 'CBT'))
    >>> cbt.acronym
    'CBT'
    >>> len(cbt.classes)
    112
    >>> len(cbt.images)
    1792
    >>> _, imgname = os.path.split(cbt.images[0])
    >>> imgname
    'D1_COLORED_1.tif'
    >>> cbt.get_class(cbt.images[0])
    'D001'

    References
    ----------
    .. [1] https://bit.ly/2LgkCnX
    
    """
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'CBT'


    def get_class(self, img):
        """Extract the class label from the given image file name."""
        head, _ = os.path.split(img)
        _, tail = os.path.split(head)
        return tail


class ForestMacro(TextureDataset):
    """Class for ForestMacro dataset.

    Notes
    -----  
    The Forest Species Database - Macro is composed of 2942 macroscopic 
    images from 41 different forest species (classes) of the Brazilian flora. 
    There are between 37 and 99 samples per class. The database was 
    collected using a Sony DSC T20 with the macro function activated. 
    The resulting images are then saved in JPG format with no compression.
    
    Image format : .jpg (RGB)
    Sample size : 2448x3264 px

    Examples
    --------
    >>> import config
    >>> foma = ForestMacro(os.path.join(config.imgs, 'ForestMacro'))
    >>> foma.acronym
    'ForestMacro'
    >>> len(foma.classes)
    41
    >>> len(foma.images)
    2942
    >>> n = 8
    >>> _, imgname = os.path.split(foma.images[n])
    >>> imgname
    '0109.JPG'
    >>> foma.get_class(foma.images[n])
    '01'

    References
    ----------
    .. [1] https://bit.ly/2LqVs6b

    """
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'ForestMacro'


    def get_class(self, img):
        """Extract the class label from the given image file name."""
        _, filename = os.path.split(img)
        return filename[:2]


class ForestMicro(TextureDataset):
    """Class for ForestMicro dataset.

    Notes
    -----
    The Forest Species Database - Micro comprehends 2240 microscopic 
    images representing samples from 112 species (classes) from 2 groups 
    (hardwood and softwood), 85 genera and 30 families. There are 20 
    samples per class, numbered from 1 to 20. The images were acquired 
    from the sheets of wood using a Olympus Cx40 microscope with 
    100 times zoom.

    Image format : .png (RGB)
    Sample size : 768x1024 px
    Sample file name : Hardwood/046 Copaifera trapezifolia/04605.png
        class : 046 Copaifera trapezifolia
        sample : 05

    Examples
    --------
    >>> import config
    >>> fomi = ForestMicro(os.path.join(config.imgs, 'ForestMicro'))
    >>> fomi.acronym
    'ForestMicro'
    >>> len(fomi.classes)
    112
    >>> len(fomi.images)
    2240
    >>> n = 164
    >>> _, imgname = os.path.split(fomi.images[n])
    >>> imgname
    '04605.png'
    >>> fomi.get_class(fomi.images[n])
    '046 Copaifera trapezifolia'

    References
    ----------
    .. [1] https://bit.ly/2Y6pUIo
    """
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'ForestMicro'


    def get_class(self, img):
        """Extract the class label from the given image file name."""
        head, _ = os.path.split(img)
        _, tail = os.path.split(head)
        return tail


class Kather(TextureDataset):
    """Class for Kather dataset.

    Notes
    -----
    Kather consists of 5,000 histological images of human colorectal cancer
    including 8 different types of tissue (625 samples per class).

    Image format : .tif (RGB)
    Sample size : 150x150 px

    Sample file name : 01_TUMOR/1A11_CRC-Prim-HE-07_022.tif_Row_601_Col_151.tif
        class : 01_TUMOR

    Examples
    --------
    >>> import config
    >>> kat = Kather(os.path.join(config.imgs, 'Kather'))
    >>> kat.acronym
    'Kather'
    >>> len(kat.classes)
    8
    >>> len(kat.images)
    5000
    >>> n = 0
    >>> _, imgname = os.path.split(kat.images[n])
    >>> imgname
    '10009_CRC-Prim-HE-03_009.tif_Row_301_Col_151.tif'
    >>> kat.get_class(kat.images[n])
    '01_TUMOR'
    
    References
    ----------
    .. [1] Multi-class texture analysis in colorectal cancer histology
           Jakob Nikolas Kather, Cleo-Aron Weis, Francesco Bianconi,
           Susanne M. Melchers, Lothar R. Schad, Timo Gaiser, Alexander Marx
           and Frank Gerrit Zöllner
           https://www.nature.com/articles/srep27988
           
    """
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'Kather'
        

    def get_class(self, img):
        """Extract the class label from the given image file name."""
        head, _ = os.path.split(img)
        _, tail = os.path.split(head)
        return tail


class KTHTIPS2b(TextureDataset):
    """Class for KTH-TIPS-2b dataset.

    The KTH-TIPS-2b dataset contains 11 types of materials such as bread,
    cotton, alluminium foil, etc. There are 4 samples per class. Each
    material sample was acquired under 9 different scales, 3 viewpoints
    and 4 illumination directions. As a result there are 432 samples
    per class (4752 samples in total).

    Image format : .png (RGB)
    Sample size : 85-299 x  71-291 px (range of nrows x range of ncols)
                  147x75 - 299x139 px (range of number of pixels)

    Sample file name : aluminium_foil/sample_d/15d-scale_2_im_1_col.png
        class : aluminium_foil
        corresponding CURET sample number: 15
        sample : d
        scale number : 2
        image number : 1 =>
            object pose : frontal
            illumination direction: frontal

    Examples
    --------
    >>> import config
    >>> kth = KTHTIPS2b(os.path.join(config.imgs, 'KTHTIPS2b'))
    >>> kth.acronym
    'KTH2b'
    >>> len(kth.classes)
    11
    >>> len(kth.images)
    4752
    >>> n = 0
    >>> _, imgname = os.path.split(kth.images[n])
    >>> imgname
    '15a-scale_10_im_10_col.png'
    >>> kth.get_class(kth.images[n])
    'aluminium_foil'
    >>> kth.get_scale(kth.images[n])
    10
    >>> kth.get_viewpoint(kth.images[n])
    'Frontal'
    >>> kth.get_illumination(kth.images[n])
    'Ambient'

    References
    ----------
    .. [1] www.nada.kth.se/cvap/databases/kth-tips
    
    """
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.scales = np.asarray([self.get_scale(img) for img in self.images])
        self.viewpoints = np.asarray(
                [self.get_viewpoint(img) for img in self.images])
        self.illuminations = np.asarray(
                [self.get_illumination(img) for img in self.images])
        self.acronym = 'KTH2b'


    def get_class(self, img):
        """Extract the class label from the given image file name."""
        subfolder = os.path.dirname(img)
        folder, _ = os.path.split(subfolder)
        return os.path.split(folder)[-1]


    def get_scale(self, img):
        """Extract the scale number from the given image file name."""
        _, filename = os.path.split(img)
        _, tail = filename.split('-')
        return int(tail.split('_')[1])


    def get_viewpoint(self, img):
        """Extract the viewpoint from the given image file name."""
        _, filename = os.path.split(img)
        _, tail = filename.split('-')
        img_number = tail.split('_')[3]
        if img_number in ['1', '2', '3', '10']:
            return 'Frontal'
        elif img_number in ['4', '5', '6', '11']:
            return '22.5 Right'
        elif img_number in ['7', '8', '9', '12']:
            return '22.5 Left'
        else:
            raise ValueError('Incorrect image number')

            
    def get_illumination(self, img):
        """Extract the illumination direction from the image file name."""        
        _, filename = os.path.split(img)
        _, tail = filename.split('-')
        img_number = tail.split('_')[3]
        if img_number in ['1', '4', '7']:
            return 'Frontal'
        elif img_number in ['2', '5', '8']:
            return '45 top'
        elif img_number in ['3', '6', '9']:
            return '45 side'
        elif img_number in ['10', '11', '12']:
            return 'Ambient'
        else:
            raise ValueError('Incorrect image number') 


class KylbergSintorn(TextureDataset):
    """Class for the Kylberg Sintorn rotation dataset.

    Notes
    -----
    Includes 25 classes of materials such as sugar, knitwear, rice, wool, etc.
    There is 1 image for each class, which was acquired using 
    invariable illumination conditions and at 9 in-plane rotation angles. 
    The images were subdivided into 16 non-verlapping subimages, thus 
    resulting in 144 samples per class 
    (25 classes x 1 image/(class & rotation) x 9 rotations x 16 samples/image
    = 3600 samples in total.
    
    Image format : .png (RGB + Transparency)
    Sample size : 864x1296 px
                
    Sample file name : canesugar01-r000-s001.png
        class : canesugar01
        rotation : 000 degrees
        sample : 001

    Examples
    --------
    >>> import config
    >>> kyl = KylbergSintorn(os.path.join(config.imgs, 'KylbergSintorn'))
    >>> kyl.acronym
    'KylSin'
    >>> len(kyl.classes)
    25
    >>> len(kyl.images)
    3600
    >>> n = 0
    >>> _, imgname = os.path.split(kyl.images[n])
    >>> imgname
    'canesugar01-r000-s001.png'
    >>> kyl.get_class(kyl.images[n])
    'canesugar01'
    >>> kyl.get_rotation(kyl.images[n])
    0

    References
    ----------
    .. [1] http://www.cb.uu.se/~gustaf/KylbergSintornRotation/

    """
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.rotations = np.asarray(
                [self.get_rotation(img) for img in self.images])
        self.acronym = 'KylSin'


    def get_class(self, img):
        """Extract the class label from the given image file name."""        
        _, fname = os.path.split(img)
        return fname.split('-')[0]


    def get_rotation(self, img):
        """Extract the rotation angle from the given image file name."""
        _, fname = os.path.split(img)
        return int(fname.split('-')[1][1:])


class MondialMarmi20(TextureDataset):
    """Class for MondialMarmi20 texture dataset.

    Notes
    -----
    MondialMarmi20 comprises 25 classes of marble and granite products 
    identified by their commercial denominations, e.g. Azul Platino, 
    Bianco Sardo, Rosa Porriño and Verde Bahía. Each class is represented 
    by 4 tiles; 10 images for each tile were acquired under steady 
    illumination conditions and at rotation angles from 0 to 90 degrees 
    by steps of 10 degrees.

    Image format : .bmp (RGB)
    Sample size : 1500x1500 px

    Sample file name : AcquaMarina_A_00_01.bmp
        class : AquaMarina_A
        rotation : 00
        sample : 01
    
    Examples
    --------
    >>> import config
    >>> mm2 = MondialMarmi20(os.path.join(config.imgs, 'MondialMarmi20'))
    >>> mm2.acronym
    'Mond20'
    >>> len(mm2.classes)
    25
    >>> len(mm2.images)
    1000
    >>> n = 0
    >>> _, imgname = os.path.split(mm2.images[n])
    >>> imgname
    'AcquaMarina_A_00_01.bmp'
    >>> mm2.get_class(mm2.images[n])
    'AcquaMarina_A'
    >>> mm2.get_rotation(mm2.images[n])
    0
    
    References
    ----------
    .. [1] http://dismac.dii.unipg.it/mm/ver_2_0/index.html

    """    
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.rotations = np.array([self.get_rotation(x) for x in self.images])
        self.acronym = 'Mond20'

        
    def get_class(self, img):
        """Extract the class label from the given image file name."""        
        _, filename = os.path.split(img)
        return filename[:-10]

                            
    def get_rotation(self, img):
        """Extract the rotation angle from the given image file name."""
        _, filename = os.path.split(img)
        return int(filename[-9:-7])
    
    
class NewBarkTex(TextureDataset):
    """Class for New BarkTex dataset.
    
    Notes
    -----
    New BarkTex is a collage of different types of tree bark derived from 
    the BarkTex database. This dataset includes 6 classes with 68 images 
    per class, each of them is split into 4 sub-images, resulting in a 
    total of 1632 samples.
        
    Image format : .bmp (RGB)
    Sample size : 64x64 px

    Examples
    --------
    >>> import config
    >>> nbt = NewBarkTex(os.path.join(config.imgs, 'NewBarkTex'))
    >>> nbt.acronym
    'NewBarkTex'
    >>> for classname in nbt.classes.keys(): print(classname)
    BetulaPendula
    FagusSilvatica
    PiceaAbies
    PinusSilvestris
    QuercusRobus
    RobiniaPseudacacia
    >>> _, imgname = os.path.split(nbt.images[0])
    >>> len(nbt.images)
    1632
    >>> imgname
    '000000.bmp'
                
    References
    ----------
    .. [1] https://bit.ly/2G6WH68
    
    """
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'NewBarkTex'


    def get_class(self, img):
        """Returns the class of the given image."""
        path, _ = os.path.split(img)
        _, folder = os.path.split(path)
        return folder

    
class Outex13(TextureDataset):
    """Class for the Outex13 dataset.
         
    Outex13 consists of 68 classes. The images were acquired under invariable 
    illumination conditions (rotation 0 degrees, illuminant INCA, resolution 
    100 dpi) and subdivided into 20 non-overlapping subimages, resulting in a 
    total number of samples of 1360.

    Image format : .bmp (RGB)
    Sample size : 128x128 px

    Sample file name : c01/000000.bmp
        class : c01
        sample : 000000

    Examples
    --------
    >>> import config
    >>> out = Outex13(os.path.join(config.imgs, 'Outex13'))
    >>> out.acronym
    'Outex13'
    >>> len(out.classes)
    68
    >>> len(out.images)
    1360
    >>> n = 0
    >>> _, imgname = os.path.split(out.images[n])
    >>> imgname
    '000000.bmp'
    >>> out.get_class(out.images[n])
    'c01'

    References
    ----------
    .. [1] https://bit.ly/2Yal8tq

    """     
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'Outex13'
        
        
    def get_class(self, img):
        """Extract the class label from the given image file pathname."""        
        head, _ = os.path.split(img)
        return os.path.split(head)[-1]


class PapSmear(TextureDataset):
    """Class for PapSmear dataset.
                
    Notes
    -----
    PapSmear consists of 917 images of Pap-smear cells of seven classes, 
    that are grouped into two classes, namely classes 1-3 are normal cells 
    (242 samples), and classes 4-7 are abnormal cells (675 samples).
        
    Image format : .bmp (RGB)
    Sample size : 32-409 x  40-768 px (range of nrows x range of ncols) 
                  43x45 - 300x768 px (range of number of pixels)
    Sample name : 3--Normal--Columnar epithelial/153956040-153956058-001.BMP
        class : Normal for two-class classification
                Columnar epithelial (3) for seven-class classification

    Examples
    --------
    >>> import config
    >>> ps2 = PapSmear(os.path.join(config.imgs, 'PapSmear'))
    >>> ps2.acronym
    'Pap2'
    >>> ps7 = PapSmear(os.path.join(config.imgs, 'PapSmear'), n_classes=7)
    >>> ps7.acronym
    'Pap7'
    >>> for name, index in ps2.classes.items(): print(f"'{name}': {index}")
    'Abnormal': 0
    'Normal': 1
    >>> for name, index in ps7.classes.items(): print(f"'{name}': {index}")
    'Columnar epithelial': 0
    'Intermediate squamous epithelial': 1
    'Mild squamous non-keratinizing dysplasia': 2
    'Moderate squamous non-keratinizing dysplasia': 3
    'Severe squamous non-keratinizing dysplasia': 4
    'Squamous cell carcinoma in situ intermediate': 5
    'Superficial squamous epithelial': 6
    >>> len(ps2.images)
    917
    >>> n = 0
    >>> head, imgname = os.path.split(ps2.images[n])
    >>> imgname
    '153958345-153958392-001.BMP'
    >>> _, foldername = os.path.split(head)
    >>> foldername
    '1--Normal--Superficial squamous epithelial'
    >>> ps2.get_class(ps2.images[n])
    'Normal'
    >>> ps7.get_class(ps7.images[n])
    'Superficial squamous epithelial'
    >>> try:
    ...     ps5 = PapSmear(os.path.join(config.imgs, 'PapSmear'), n_classes=5)
    ... except ValueError:
    ...     pass
    ...
    ValueError: Invalid number of classes: 5
    Use n_classes=2 or n_classes=7
    
    References
    ----------
    .. [1] Pap-smear Benchmark Data For Pattern Classification
           Jan Jantzen, Jonas Norup, George Dounias, Beth Bjerregaard
           https://bit.ly/2r7gBpU
    
    """
    def __init__(self, dirpath, n_classes=2):
        if n_classes not in (2, 7):
            print (f'ValueError: Invalid number of classes: {n_classes}\n'
                   f'Use n_classes=2 or n_classes=7')
            raise ValueError
        self.n_classes = n_classes
        super().__init__(dirpath)
        self.acronym = f'Pap{self.n_classes}'

    def get_class(self, img):
        """Returns the class of the given img."""
        location, _ = os.path.split(img)
        _, folder = os.path.split(location)
        
        if self.n_classes == 2:
            return folder.split('--')[1]
        elif self.n_classes == 7:
            return folder.split('--')[-1]


class Parquet(TextureDataset):
    """Class for Parquet dataset.
     Notes
    -----
    Parquet comprehends 14 commercial varieties of finished wood for 
    flooring and cladding. Each variety has between 2 and 4 grades, 
    which are considered as independent classes, yielding a total of 
    38 classes. Each class is represented by between 6 and 8 images. Each 
    image was subdivided into 4 non-ovelapping subimages. The total number 
    of samples is 1180.
            
    Image format : .bmp (RGB)
    Sample size : 240-650 x  600-1000 px (range of nrows x range of ncols) 
                  240x650 - 650x750 px (range of number of pixels)

    Sample file name : IRK_01_Grade_1__2_3.bmp
        class : IRK_01_Grade_1
        image : 2
        subimage : 3

    Examples
    --------
    >>> import config
    >>> par = Parquet(os.path.join(config.imgs, 'Parquet'))
    >>> par.acronym
    'Parquet'
    >>> len(par.classes)
    38
    >>> len(par.images)
    1180
    >>> n = 0
    >>> _, imgname = os.path.split(par.images[n])
    >>> imgname
    'IRK_01_Grade_1__1_1.bmp'
    >>> par.get_class(par.images[n])
    'IRK_01_Grade_1'

    References
    ----------
    .. [1] http://dismac.dii.unipg.it/parquet/

    """    
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'Parquet'

        
    def get_class(self, img):
        """Extract the class label from the given image file name."""        
        location, _ = os.path.split(img)
        _, folder = os.path.split(location)
        return folder


class PlantLeaves(TextureDataset):
    """Class for PlantLeaves dataset.
                
    Notes
    -----
    PlantLeaves consists of 1200 images of leaves of 20 plant species 
    (20 classes x 20 leaves/class x 3 samples/leave). Class labels go 
    from 'c01' through 'c20'.
        
    Image format : .png (RGB)
    Sample size : 128x128 px
                
    Sample file name : c017/c01_011_a_w03.png
        class : c01
        leave : 011
        sample : 03

    Examples
    --------
    >>> import config
    >>> pl = PlantLeaves(os.path.join(config.imgs, 'PlantLeaves'))
    >>> pl.acronym
    'Plant'
    >>> len(pl.classes)
    20
    >>> len(pl.images)
    1200
    >>> n = 50
    >>> head, imgname = os.path.split(pl.images[n])
    >>> imgname
    'c01_017_a_w03.png'
    >>> _, foldername = os.path.split(head)
    >>> foldername
    'c01'
    >>> pl.get_class(pl.images[n])
    'c01'
    
    References
    ----------
    .. [1] Plant Leaf Identification Using Gabor Wavelets
           Dalcimar Casanova, Jarbas Joaci de Mesquita Sá Junior, Odemir 
           Martinez Bruno
           https://doi.org/10.1002/ima.20201

    """
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'Plant'


    def get_class(self, img):
        """Returns the class of the given img."""
        location, _ = os.path.split(img)
        _, folder = os.path.split(location)
        return folder


class STex(TextureDataset):
    """Class for STex dataset.
    
    The Salzburg Texture Image Database is a collection of 476 colour 
    texture images acquired 'in the wild' around Salzburg, Austria. They 
    mainly represent objects and materials like bark, floor, leather, etc. 
    The dataset comes into two different resolutions, namely 1024x1024 px 
    and 512x512 px, of which the second was the one we use. We further 
    subdivided the original images into 16 non-overlapping subimages of 
    dimensions 128x128 px, resulting in 7616 samples in total.

    Image format : .bmp (RGB)
    Sample size : 128x128 px

    Sample file name : Bark_0004_13.bmp
        class : Bark_0004
        subimage : 13

    Examples
    --------
    >>> import config
    >>> stex = STex(os.path.join(config.imgs, 'STex'))
    >>> stex.acronym
    'STex'
    >>> len(stex.classes)
    476
    >>> len(stex.images)
    7616
    >>> n = 100
    >>> _, imgname = os.path.split(stex.images[n])
    >>> imgname
    'Bark_0006_05.bmp'
    >>> stex.get_class(stex.images[n])
    'Bark_0006'
    
    References
    ----------
    .. [1] http://www.wavelab.at/sources/STex/

    """            
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'STex'


    def get_class(self, img):
        """Returns the class of the given img."""        
        folder, _ = os.path.split(img)
        return os.path.split(folder)[-1]


class VxCTSG(TextureDataset):
    """Class for VxCTSG dataset.
    
    The VxC TSG dataset is a collection of 14 classes representing 
    commercial denominations of ceramic tiles with 3 grades per class. We 
    considered each grade as a class on its own, which gives 42 classes 
    in total. The images were acquired in laboratory under controlled and 
    invariable conditions. The number of samples per class varies 
    from 14 to 30, resulting in 504 samples in total.
    
    Image format : .bmp (RGB)
    Sample size : from 500x500 px to 950x950 px

    Sample file name : berlin-grade11/05.bmp
        class : berlin-grade11
        subimage : 05

    Examples
    --------
    >>> import config
    >>> vxc = VxCTSG(os.path.join(config.imgs, 'VxCTSG'))
    >>> vxc.acronym
    'VxCTSG'
    >>> len(vxc.classes)
    42
    >>> len(vxc.images)
    504
    >>> n = 100
    >>> folder, imgname = os.path.split(vxc.images[n])
    >>> os.path.split(folder)[-1]
    'berlin-grade11'
    >>> imgname
    '05.bmp'
    >>> vxc.get_class(vxc.images[n])
    'berlin-grade11'
    
    References
    ----------
    .. [1] Performance evaluation of soft color texture descriptors for 
           surface grading using experimental design and logistic regression
           Fernando López, José Miguel Valiente, José Manuel Prats, 
           and Alberto Ferrer
           https://bit.ly/2YQfYQx
           http://miron.disca.upv.es/vision/vxctsg/ (broken link)
           
    """ 
    def __init__(self, dirpath):
        super().__init__(dirpath)
        self.acronym = 'VxCTSG'


    def get_class(self, img):
        """Returns the class of the given img."""        
        folder, _ = os.path.split(img)
        return os.path.split(folder)[-1]


################
# Useful stuff #
################

def subdivide_CBT(source, destination, x=4, y=4):
    """Utility function for subdividing the RGB images from the
    Coloured Brodatz Textures dataset.

    Parameters
    ----------
    source : string
        Full path of the folder where the original images are stored.
    destination : string
        Full path of the folder where the subimages will be saved.
    x : int
        Number of vertical subdivisions.
    y : int
        Number of horizontal subdivisions.
        
    """
    from skimage import io
    from IPython.utils.path import ensure_dir_exists

    ensure_dir_exists(destination)
    
    for n in range(1, 113):
        imgpath = os.path.join(source, f'D{n}_COLORED.tif')
        img = io.imread(imgpath)
        folder = os.path.join(destination, f'D{n:03}')
        ensure_dir_exists(folder)
        rows = np.int_(np.linspace(0, img.shape[0], x + 1))
        cols = np.int_(np.linspace(0, img.shape[1], y + 1))
        for i, (start_row, end_row) in enumerate(zip(rows[:-1], rows[1:])):
            for j, (start_col, end_col) in enumerate(zip(cols[:-1], cols[1:])):
                sub = img[start_row:end_row, start_col:end_col, :]
                sample = 1 + i*y + j
                subname = f'D{n}_COLORED_{sample}.tif'
                subpath = os.path.join(folder, subname)
                io.imsave(subpath, sub)


def download_KylbergSintorn(destination, x=4, y=4):
    """Utility function to download Kylberg Sintorn Rotation Dataset and
    subdivide the hardware-rotated images.
    
    Parameters
    ----------
    destination : string
        Full path of the folder where the subimages will be saved.
    x : int
        Number of vertical subdivisions.
    y : int
        Number of horizontal subdivisions.

    """
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    import re
    from skimage import io

    url = 'http://www.cb.uu.se/~gustaf/KylbergSintornRotation/data/png-originals/'
    html = urlopen(url)
    soup = BeautifulSoup(html.read())
    for link in soup.find_all('a', href=re.compile('.+png$')):
        filename = link['href']
        fn, ext = os.path.splitext(filename)
        img = io.imread(url + '/' + filename)
        rows = np.int_(np.linspace(0, img.shape[0], x + 1))
        cols = np.int_(np.linspace(0, img.shape[1], y + 1))
        for i, (start_row, end_row) in enumerate(zip(rows[:-1], rows[1:])):
            for j, (start_col, end_col) in enumerate(zip(cols[:-1], cols[1:])):
                sub = img[start_row:end_row, start_col:end_col, :]
                sample = 1 + i*y + j
                subname = f'{fn}-s{sample:03}{ext}'
                subpath = os.path.join(destination, subname)
                io.imsave(subpath, sub)


def subdivide_Parquet(source='', destination='', x=2, y=2):
    """Utility function to subdivide the RGB images from the Parquet database.

    Parameters
    ----------
    source : string
        Full path of the folder where the original RGB images are stored.
    destination : string
        Full path of the folder where the subimages will be saved.
    x : int
        Number of vertical subdivisions.
    y : int
        Number of horizontal subdivisions.
        
    """
    #source = r'D:\mydatadrive\Datos\Texture datasets\Texture databases\Parquet'
    #destination = r'C:\texture\images\Parquet'
    from skimage import io
    from IPython.utils.path import ensure_dir_exists

    original = [os.path.join(root, filename)
                for root, dirs, files in os.walk(source)
                for filename in files
                if imghdr.what(os.path.join(root, filename))]

    ensure_dir_exists(destination)
    for o in original:
        img = io.imread(o)
        _, file_source = os.path.split(o)
        name, ext = os.path.splitext(file_source)
        klass, sample = name.split('__')
        ensure_dir_exists(os.path.join(destination, klass))

        rows = np.int_(np.linspace(0, img.shape[0], x + 1))
        cols = np.int_(np.linspace(0, img.shape[1], y + 1))
        for i, (start_row, end_row) in enumerate(zip(rows[:-1], rows[1:])):
            for j, (start_col, end_col) in enumerate(zip(cols[:-1], cols[1:])):
                sub = img[start_row:end_row, start_col:end_col, :]
                sample = 1 + i*y + j
                subname = f'{name}_{sample}{ext}'
                subpath = os.path.join(destination, klass, subname)
                io.imsave(subpath, sub)

    
if __name__ == '__main__':
    
    import doctest
    doctest.testmod()