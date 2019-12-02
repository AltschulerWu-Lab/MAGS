"""
Cell-type specific segmentation pipelines
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage as ndi
from scipy import stats
import seaborn as sns
sns.set(rc={'image.cmap': u'Greys_r'})
from skimage import color, draw, feature, filters, io, measure, morphology, restoration, segmentation
import warnings

from utils import imfuns

class Segmentor:
  """
  General segmentation class

  Attributes:
    C (dict): segmentation parameters
    dreader (DataReader): contains experiment info
    im (float ndarray): image to be segmented
    iminfo (ImageInfo): location information about image
    
    blobs (bool ndarray): centers of detected blobs have value 1 (same dimension as image)
    segmentation (label ndarray): labeled objects from segmentation
    
    seg_outpath (str): path to save segmentation file
    overlay_outpath (str): path to save segoverlay
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    """
    Loads attributes

    Args:
      dreader (DataReader): for current plate
      iminfo (ImageInfo): for image to be segmented
      im (float ndarray): image
      im_smooth (float ndarray): smoothed image
      seg_params (dict): segmentation params 
    """

    if im.dtype != np.dtype('float'):
      raise TypeError('Image not is not a float array')

    self.dreader = dreader
    self.im = im
    self.iminfo = iminfo

    self.object_type = None

    self.C = seg_params

    # store processing output
    self.blobs = []
    self.im_segmented = []
    self.im_smooth = []

    # output file paths
    self.seg_outpath = ''
    self.overlay_outpath = ''

  def bounded_thresh(self, potential_thresh):
    """
    A potential threshold is lowered (THRESHOLD_DOWN_FACTOR). If when adjusted, it is lower than a 
    factor of the image pixel mode (THRESHOLD_MODE_FACTOR), the factor of image mode is used instead

    Args:
      potential_threshold (float): threshold for image

    Returns:
      float: adjusted threshold for image
    """

    im_pxmode = stats.mode(self.im, axis=None).mode[0]

    if im_pxmode == 1:
      return max(self.C['THRESHOLD_DOWN_FACTOR']*potential_thresh, self.C['THRESHOLD_MODE_FACTOR_HI']*im_pxmode)
    else:
      return max(self.C['THRESHOLD_DOWN_FACTOR']*potential_thresh, self.C['THRESHOLD_MODE_FACTOR']*im_pxmode)

  def denoise_image(self):
    """
    Abstracted version of denoise_bilateral function. Runs function on raw image using given constants
    """
    return restoration.denoise_bilateral(self.im, sigma_color=self.C['BILATERAL_SIGMA_COLOR'], 
      sigma_spatial=self.C['BILATERAL_SIGMA_SPATIAL'], multichannel=False)

  def draw_circles(self):
    """
    Converts detected blob centers into circles at those centers on the segmentation image
    """

    for num, blob in enumerate(self.blobs):
    
      y, x, r = blob
      rr, cc = draw.circle(y, x, r)
      rc_filt = [(r, c) for r, c in zip(rr, cc) if 0 <= r < self.im_segmented.shape[0] and 
        0 <= c < self.im_segmented.shape[1]]
      im_rr, im_cc = zip(*rc_filt)
      self.im_segmented[im_rr,im_cc] = num+1

  def blobs_to_markers(self, im_shape, blobs):
    """
    Converts output of blob detection to matrix containing center of blobs (seeds)  

    Args:
      im_shape (tuple): dimension of image (e.g. (2,3) for 2 by 3 pixel image)
      blobs (list of tuples): list of (x, y, ~, ...) indicating x, y position of 
        blob centers

    Returns:
      labeled ndarray: image with labeled center positions (size as provided)

    """

    im_seeds = np.zeros(im_shape)
    im_seeds[blobs[:,0].astype(int), blobs[:,1].astype(int)] = 1

    markers, markers_num = ndi.label(im_seeds)

    return markers

  def filter_median(self, im, filter_size):
    """
    Abstracted media filter function that catches warnings about image type conversion (float to uint8)

    Args:
      im (ndarray): image
      filter_size (int): size of disk filter

    Returns:
      uint8 ndarray: filtered image
    """

    with warnings.catch_warnings():

      # catches warnings (e.g. about labeled)
      warnings.simplefilter("ignore")

      return filters.median(im, selem=morphology.disk(filter_size))

  def find_blobs(self, im, p):
    """
    Abstracted version of blob_log function

    Args: 
      im (ndarray): input image
      p (dict): parameters for function (must include keys 'MIN_SIG', 'MAX_SIG', 
        'NUM_SIG', 'THRESH', and 'OVERLAP')

    Returns:
      list of tuples: [(x, y, r)] where x and y are coordinates of blob center and 
        r is radius of blob
    """

    blobs = feature.blob_log(im, min_sigma=p['MIN_SIG'], max_sigma=p['MAX_SIG'], 
      num_sigma=p['NUM_SIG'], threshold=p['THRESH'], overlap=p['OVERLAP'])

    blobs[:, 2] = blobs[:, 2] * math.sqrt(2)

    return blobs

  def label2rgb(self, im_labeled, im=None):
    """
    Abstracted version of label2rgb

    Args:
      im_labeled (labeled ndarray): regions to false color

    Returns:
      rbg ndarray: colored regions overlay on image
    """

    if im is None:
      im = self.im

    return color.label2rgb(im_labeled, image=im, bg_label=0)

  def plot_markers(self, ax, im, markers, color='yellow'):
    """
    Plots markers (seeds) for segmentation on image

    Args:
      ax (axis): plot to use
      im (ndarray): image for background
      markers (bool/label ndarray): single pixels to indicate marker location

    Returns:
      Result is plotted on given plot axis
    """
    ax.imshow(im, zorder=1)

    xs, ys = np.where(markers > 0 )

    for x,y in zip(list(xs), list(ys)):
      c = plt.Circle((y, x), 1, color=color, linewidth=1, fill=True, zorder=2)
      ax.add_patch(c)

  def plot_results(self, save=False, show=True):

    fig, axes = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(self.im, zorder=1)
    ax[0].set_title('Raw Image')
    ax[1].imshow(self.im_smooth, zorder=1)
    ax[1].set_title('Smoothed Image')
    ax[2].imshow(self.im, zorder=1)
    for blob in self.blobs:
      y, x, r = blob
      c = plt.Circle((x, y), 1, color='yellow', linewidth=1, fill=True, zorder=2)
      ax[2].add_patch(c)
    ax[2].set_title('Seeds')
    ax[3].imshow(self.label2rgb(self.im_segmented), zorder=1)
    ax[3].set_title('Segmentation')

    if save:
      outpath = self.dreader.segviz_impath.get_path(self.iminfo).format(object_type=self.object_type)
      fig.savefig(outpath)
      
    if show:
      plt.show()

    plt.close()

  def process_tophat(self, im_denoise):
    """
    Process image using tophat, closing, then dilation
    """
    im_wtophat = morphology.white_tophat(im_denoise, selem=morphology.disk(self.C['WTOPHAT_FILTER_SZ']))
    im_closed = morphology.closing(im_wtophat)
    im_processed = morphology.dilation(im_closed)

    return im_processed

  def remove_markers(self, markers, markers_unwanted, kernel_size):
    """
    Removes markers close to unwanted markers (in case detected location is slightly different) 

    Args:
      markers (ndarray): int label at seed positions, 0 elsewhere
      markers_unwanted (ndarray): int label at seed positions, 0 elsewhere, same size as markers
      kernel_size (int): size of dilation operation (essentially distance to unwanted markers)

    Returns:
      labeled ndarray: labeled markers with unwanted markers removed

    """
    if markers.shape != markers_unwanted.shape:
      raise ValueError('Input matrices should be same shape')

    markers_unwanted_expanded = morphology.dilation(markers_unwanted, selem=morphology.disk(kernel_size))

    markers_filtered = np.copy(markers)
    markers_filtered[markers_unwanted_expanded!=0] = 0

    return markers_filtered

  def remove_small_holes(self, im, min_size):
    """
    Abstracted remove_small_holes function that catches the warnings about labeled arrays

    Args:
      im (labeled/bool ndarray): image with holes
      min_size (int): size below which to remove holes

    Returns:
      bool ndarray: labeled ndarray inputs will be converted into bool ndarrays
    """

    with warnings.catch_warnings():

      # catches warnings (e.g. about labeled)
      warnings.simplefilter("ignore")
      
      return morphology.remove_small_holes(im, min_size=min_size)

  def remove_small_objects(self, im, min_size):
    """
    Abstracted remove_small_objects function that catches the warnings about labeled arrays

    Args:
      im (labeled/bool ndarray): image with holes
      min_size (int): size below which to remove holes

    Returns:
      bool ndarray: labeled ndarray inputs will be converted into bool ndarrays
    """

    with warnings.catch_warnings():

      # catches warnings (e.g. about labeled)
      warnings.simplefilter("ignore")
      
      return morphology.remove_small_objects(im, min_size=min_size)

  def segment_circle(self, default_params=True):
    """
    'Segmentation' based on scale-space Laplacian of Gaussian blob detection of cells. Determines 
    number and location of cells. Determining cell boundaries are not attempted

    Args: 
      default_params (bool): if true, self.C['LOG_BLOB'] are used for log function params
    """
    
    if default_params is True:
      params = self.C['LOG_BLOB']
    else:
      params = default_params

    self.blobs = self.find_blobs(self.im_smooth, params)

    self.im_segmented = np.zeros(self.im.shape, dtype=np.uint16)

    self.draw_circles()

    markers = self.blobs_to_markers(self.im.shape, self.blobs)

    return markers

  def segment_watershed(self, im, im_thresh, compact=True, line=False, default_params=True):
    """
    Segmentation by first detecting cell locations using scale-space Laplacian of Gaussian blob 
    detection. Cell boundaries are determined using watershed

    Args: 
      default_params (bool): if true, self.C['LOG_BLOB'] are used for log function params

    Returns:
      labeled ndarray: segmented objects
    """

    if default_params is True:
      params = self.C['LOG_BLOB']
    else:
      params = default_params

    self.blobs = self.find_blobs(im_thresh, params)

    markers = self.blobs_to_markers(im.shape, self.blobs)

    im_segmented = self.watershed(im, markers, im_thresh, line=line, compact=compact)

    return markers, im_segmented

  def thresh_otsu(self, im_smooth):
    """
    Modified otsu thresholding such that if the image is blank, the 'threshold' is greater than the 
    image max intensity
    """
    try: 
      otsu_thresh = filters.threshold_otsu(im_smooth, nbins=self.C['OTSU_BINS'])
    except ValueError:
      otsu_thresh = np.max(im_smooth) + 1

    return otsu_thresh

  def thresh_twolevel(self, im_denoise):
    """
    Switch threshold depending on level of non-specific tissue background signal

    Args:
      im_denoise (float ndarray): denoised image
    """

    otsu_thresh = filters.threshold_otsu(im_denoise, nbins=self.C['OTSU_BINS'])
    im_mask = np.copy(im_denoise)
    im_mask[im_mask < otsu_thresh] = np.nan
    threshed_mean_val = np.nanmean(im_mask)

    if threshed_mean_val > self.C['THRESH_BOUNDARY']:
      return self.C['THRESH_HIGH']
    else:
      return self.C['THRESH_LOW']


  def watershed(self, im, markers, im_thresh, compact=True, line=False):
    """
    Slightly more abstracted watershed function call

    Args:
      im (ndarray): raw image
      markers (labeled ndarray): labeled seeds 
      im_thresh (ndarray): is 0 at not-cell pixels
      compact (bool): if True, use given constant. Else, use 0
      line (bool): if True, draw separating lines in output

    Returns:
      labeled ndarray: segmented image

    """

    if compact:
      compactness = self.C['WATERSHED_COMPACTNESS']
    else:
      compactness = 0

    im_inverted = ((1-im)*self.dreader.max_px_val).astype(int)

    im_watershed = segmentation.watershed(im_inverted, markers, compactness=compactness, 
      connectivity=self.C['WATERSHED_CONN'], mask=im_thresh!=0, watershed_line=line)

    return self.remove_small_objects(im_watershed, self.C['WATERSHED_MIN_SZ'])

  def preprocess(self):
    pass

  def segment(self):
    pass

  def save(self):
    """
    Saves the segmentation (labeled ndarray) image and segmentation overlay (rbg ndarray) image
    """

    seg_prefix = os.path.dirname(self.seg_outpath)
    if not os.path.exists(seg_prefix):
      os.makedirs(seg_prefix)

    overlay_prefix = os.path.dirname(self.overlay_outpath)
    if not os.path.exists(overlay_prefix):
      os.makedirs(overlay_prefix)

    with warnings.catch_warnings():

      # catches warnings (e.g. low contrast image)
      warnings.simplefilter("ignore")

      io.imsave(self.seg_outpath, np.array(self.im_segmented).astype(np.uint16))
      io.imsave(self.overlay_outpath, self.label2rgb(self.im_segmented))

  def run(self, plot=False):
    self.preprocess()
    self.segment()
    self.save()

    if plot:
      self.plot_results(save=True, show=False)

class Crypt_Finder(Segmentor):
  """
  Crypt Segmentation

  Attributes:
    im_thresh (float ndarray): thresholded image
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    Segmentor.__init__(self, dreader, iminfo, im, seg_params)

    # storing images
    self.im_thresh = []

    self.object_type = 'crypt'

    # output files
    self.seg_outpath = self.dreader.crypt_objects_impath.get_path(self.iminfo)
    self.overlay_outpath = self.dreader.crypt_overlay_impath.get_path(self.iminfo)

  def preprocess(self):
    self.threshold()

  def segment(self):
    self.im_segmented = measure.label(self.im_threshed)

  def threshold(self):
    """Threshold by first removing nuclear stain bleed through"""

    dna_im = self.dreader.get_max_im(self.iminfo, 'dna')
    im_subtracted = imfuns.subtract(self.im, self.C['DNA_FACTOR']*dna_im)
    im_mask = imfuns.imthresh(im_subtracted, self.C['THRESH']) > 0

    im_closed = morphology.binary_closing(im_mask, selem=morphology.disk(self.C['MORPH_CLOSING_SZ']))
    im_opened = morphology.binary_opening(im_closed, selem=morphology.disk(self.C['MORPH_OPENING_SZ']))
    self.im_threshed = morphology.remove_small_objects(im_opened, min_size=self.C['MIN_SZ']) 
    self.im_smooth = imfuns.mask_im(self.im, self.im_threshed)

    return im_subtracted, im_mask, im_closed, im_opened

class EE_Segmentor(Segmentor):
  """
  EE (enteroendocrine) Segmentation

  Attributes:
    im_denoise (float ndarray): noise-removed image
    im_smooth (float ndarray): smoothed image
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    """See superclass Segmentor"""
    Segmentor.__init__(self, dreader, iminfo, im, seg_params)

    # storing images
    self.im_denoise = []

    # output files
    self.seg_outpath = self.dreader.ee_objects_impath.get_path(self.iminfo)
    self.overlay_outpath = self.dreader.ee_overlay_impath.get_path(self.iminfo)

  def preprocess(self):
    self.im_denoise = self.denoise_image()
    self.im_smooth = self.process_tophat(self.im_denoise)

  def segment(self):
    self.C['LOG_BLOB']['THRESH'] = self.thresh_twolevel(self.im_denoise)
    markers = self.segment_circle()

    return markers

class Goblet_Segmentor(Segmentor):
  """
  Goblet Segmentation

  Attributes:
    im_smooth (float ndarray): smoothed image
    im_thresh (float ndarray): threshed image where under threshold has value 0, over threshold has 
      original value 
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    """
    See superclass Segmentor
    """
    Segmentor.__init__(self, dreader, iminfo, im, seg_params)

    # storing images
    self.im_thresh = []

    # output files
    self.seg_outpath = self.dreader.goblet_objects_impath.get_path(self.iminfo)
    self.overlay_outpath = self.dreader.goblet_overlay_impath.get_path(self.iminfo)

  def preprocess(self):
    self.smooth()
    self.threshold()

  def segment(self):
    markers, self.im_segmented = self.segment_watershed(self.im, self.im_thresh)

    return markers

  def smooth(self):
    """
    Smooths image using media filtering
    """

    self.im_smooth = self.filter_median(self.im, self.C['MEDIAN_FILTER_SZ'])

  def threshold(self):
    """
    Thresholds by first finding the Otsu threshold. Holes are removed from Otsu thresholded image and
    convex hulls are created for foreground objects. Foreground objects are expanded to fill the convex
    hulls.
    """
    
    otsu_thresh = self.thresh_otsu(self.im_smooth)
    thresh_val = self.bounded_thresh(otsu_thresh)
    
    # convex hull threshold result
    thresh_mask = self.im_smooth > thresh_val
    thresh_mask = ndi.binary_fill_holes(thresh_mask)
    thresh_mask = morphology.convex_hull_object(thresh_mask)

    self.im_thresh = morphology.closing(self.im_smooth, selem=morphology.disk(self.C['CLOSING_FILTER_SZ']))
    self.im_thresh[~thresh_mask] = 0

    return otsu_thresh, thresh_val

  def plot_results(self):

    fig, axes = plt.subplots(1, 3, figsize=(21,7), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(self.im, zorder=1)
    ax[0].set_title('Raw Image')
    ax[1].imshow(self.im_thresh, zorder=1)
    ax[1].set_title('Smoothed & Thresholded Image')
    ax[2].imshow(self.label2rgb(self.im_segmented), zorder=1)
    ax[2].set_title('Goblet Segmentation')

    plt.show()

class Nuclear_Segmentor(Segmentor):
  """
  Nuclear Segmentation

  Attributes:
    im_thresh (float ndarray): threshed image where under threshold has value 0, over threshold has 
      original value 
    markers_dense (labeled ndarray): labeled centers of detected dense blobs
    markers_sparse (labeled ndarray): labeled centers of detected sparse blobs
    seg_dense (labeled ndarray): dense segmentation result
    seg_sparse (labeled ndarray): sparse segmentation result
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    """
    See superclass Segmentor
    """
    Segmentor.__init__(self, dreader, iminfo, im, seg_params)

    self.object_type = 'dna'

    # storing images
    self.im_thresh = []
    self.seg_firstpass = []
    self.seg_sparse = []
    self.im_clumps = []
    self.seg_dense = []

    # output files
    self.seg_outpath = self.dreader.dna_objects_impath.get_path(self.iminfo)
    self.overlay_outpath = self.dreader.dna_overlay_impath.get_path(self.iminfo)

  def find_clumps(self):
    """
    Find clumps in segmentation image. Clumps include: all objects larger than SEG_SINGLE_MAX_SZ and
    objects between SEG_SINGLE_MIN_SZ and SEG_SINGLE_MAX_SZ that are irregular. Creates a raw image 
    masked to show only clumped regions. Filters clumps from sparse segmentation

    Args: None

    Returns: None
    """

    seg_nodebris = self.remove_small_objects(self.seg_firstpass, self.C['SEG_DEBRIS_SZ'])

    seg_large_clumps = self.remove_small_objects(seg_nodebris, self.C['SEG_SINGLE_MAX_SZ'])
    
    seg_mixed = self.remove_small_objects(seg_nodebris, self.C['SEG_SINGLE_MIN_SZ'])
    seg_mixed[seg_large_clumps!=0]=0
    seg_mixed_irregular = self.find_irregular_objects(seg_mixed, self.C['SEG_CLUMP_SOLIDITY'])

    seg_clumps = np.maximum(seg_mixed_irregular, seg_large_clumps) 
    seg_clumps = self.remove_small_holes(seg_clumps, self.C['SEG_CLOSE_HOLES'])

    self.im_clumps = np.copy(self.im)
    self.im_clumps[~seg_clumps]=0

    self.seg_sparse = np.copy(seg_nodebris)
    self.seg_sparse[self.im_clumps!=0] = 0

    return seg_nodebris, seg_large_clumps, seg_mixed, seg_mixed_irregular, seg_clumps

  def find_irregular_objects(self, im, solidity_thresh):
    """
    Identifies irregular objects in image

    Args: 
      im (labeled ndarray): image of objects
      solidity_thresh (float): cutoff of solidity value below which an object is irregular

    Returns:
      labeled ndarray: Containing only irregular objects in image
    """
    irregular_labels = [x.label for x in measure.regionprops(im) if x.solidity < solidity_thresh]
    im_irregular = np.copy(im)
    im_irregular[~np.isin(im, irregular_labels)] = 0

    return im_irregular

  def preprocess(self):
    self.threshold()

  def segment(self):
    """
    Runs first sparse segmentation, then dense segmentation. Final output combines the two results
    """

    self.segment_sparse()

    self.find_clumps()

    self.segment_dense()

    self.segment_combine()

  def segment_combine(self):
    """
    Combine sparse and dense seg
    """
    label_add = np.max(self.seg_sparse)+1
    seg_dense_relabeled = self.seg_dense + label_add
    seg_dense_relabeled[seg_dense_relabeled == label_add] = 0

    self.im_segmented = np.maximum(seg_dense_relabeled, self.seg_sparse)

  def segment_dense(self):
    """
    Performs dense segmentation by segmenting the clumps using dense seg parameters
    """
    markers, self.seg_dense = self.segment_watershed(self.im_clumps, self.im_clumps, line=True, 
      default_params=self.C['LOG_CLUMP'])    

    return markers

  def segment_sparse(self):
    """
    Performs sparse segmentation. Removes bright spot artifacts from output.
    """

    blobs_bright = self.find_blobs(self.im, self.C['LOG_BRIGHT'])
    blobs = self.find_blobs(self.im_thresh, self.C['LOG_BLOB'])

    markers_bright = self.blobs_to_markers(self.im.shape, blobs_bright)
    markers_unfiltered = self.blobs_to_markers(self.im.shape, blobs)

    markers_sparse = self.remove_markers(markers_unfiltered, markers_bright, 
      self.C['BRIGHT_SEEDS_SELEM_SZ'])

    self.markers_sparse = markers_sparse

    self.seg_firstpass = self.watershed(self.im, markers_sparse, self.im_thresh, compact=False)

    return markers_unfiltered, markers_bright, markers_sparse

  def threshold(self):
    """
    Threshold using Otsu
    """

    self.im_smooth = self.denoise_image()
    otsu_thresh = self.thresh_otsu(self.im_smooth)
    thresh_val = self.bounded_thresh(otsu_thresh)

    self.im_thresh = np.copy(self.im_smooth)
    self.im_thresh[self.im_smooth < thresh_val] = 0

    return otsu_thresh, thresh_val

  def plot_results(self, save=False, show=True):

    fig, axes = plt.subplots(2, 4, figsize=(28,14), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].set_title('Max projected image')
    ax[0].imshow(self.im, zorder=1)

    ax[1].set_title('Sparse segmentation')
    ax[1].imshow(self.label2rgb(self.seg_sparse), zorder=1)

    ax[4].set_title('Thresholded image')
    ax[4].imshow(self.im_thresh, zorder=1)

    ax[5].set_title('Dense segmentation')
    ax[5].imshow(self.label2rgb(self.seg_dense), zorder=1)

    plt.subplot(1,2,2, sharex=ax[0], sharey=ax[0])
    plt.title('Final Segmentation')
    plt.imshow(self.label2rgb(self.im_segmented), zorder=1)

    plt.suptitle('{object_type:s} segmentation'.format(object_type=self.object_type))

    if save:
      outpath = setting.paths['result'].format(object_type=self.object_type)
      fig.savefig(outpath)
      plt.close()
      
    if show:
      plt.show()


class EdU_Segmentor(Nuclear_Segmentor):
  """
  See superclass Nuclear_Segmentor
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    Nuclear_Segmentor.__init__(self, dreader, iminfo, im, seg_params)

    # output files
    self.seg_outpath = self.dreader.edu_objects_impath.get_path(self.iminfo)
    self.overlay_outpath = self.dreader.edu_overlay_impath.get_path(self.iminfo)

  def segment_sparse(self):
    markers, self.seg_firstpass = self.segment_watershed(self.im, self.im_thresh, compact=False)

    return markers

class Paneth_Segmentor(Segmentor):
  """
  Paneth Segmentation

  Attributes:
    im_denoise (float ndarray): noise-removed image
    im_smooth (float ndarray): smoothed image
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    """See superclass Segmentor"""
    Segmentor.__init__(self, dreader, iminfo, im, seg_params)

    # storing images
    self.im_denoise = []

    # output files
    self.seg_outpath = self.dreader.paneth_objects_impath.get_path(self.iminfo)
    self.overlay_outpath = self.dreader.paneth_overlay_impath.get_path(self.iminfo)

  def preprocess(self):
    self.im_denoise = self.denoise_image()
    self.im_smooth = self.process_tophat(self.im_denoise)
    
  def segment(self):
    self.C['LOG_BLOB']['THRESH'] = self.thresh_twolevel(self.im_denoise)
    markers = self.segment_circle()

    return markers

class Single_Lgr5_Segmentor(Crypt_Finder):
  """
  Single Lgr5 segmentation

  Attributes:
    crypt_mask (bool ndarray): value 1 if crypt region, 0 otherwise
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    """
    See superclass Crypt_Finder
    """
    Crypt_Finder.__init__(self, dreader, iminfo, im, seg_params)

    crypt_objects = dreader.crypt_objects_impath.readim(iminfo)
    self.crypt_mask = crypt_objects != 0

    # output files
    self.seg_outpath = self.dreader.singlelgr5_objects_impath.get_path(self.iminfo)
    self.overlay_outpath = self.dreader.singlelgr5_overlay_impath.get_path(self.iminfo)

  def crypt_filter(self, blobs):
    """
    Remove detected cells that are in crypt zones
    """
    return [(y,x,r) for y,x,r in blobs if self.crypt_mask[y.astype(int), x.astype(int)] == 0]

  def preprocess(self):
    self.smooth()

  def segment(self):
    """
    Detect blob centers and remove those in crypt zones. 
    """

    blobs = self.find_blobs(self.im_smooth, self.C['LOG_BLOB'])
    self.blobs = self.crypt_filter(blobs)

    self.im_segmented = np.zeros(self.im.shape, dtype=np.uint16)

    Segmentor.draw_circles(self)

  def smooth(self):
    """Smooth using median filter and close holes"""

    self.im_smooth = self.filter_median(self.im, self.C['MEDIAN_FILTER_SZ'])
    self.im_smooth = morphology.closing(self.im_smooth, selem=morphology.disk(self.C['MORPH_CLOSING_SZ']))

class Stem_Segmentor(Crypt_Finder):
  """
  Stem cell segmentation

  Attributes:
    crypt_mask (bool ndarray): value 1 if crypt region, 0 otherwise
  """

  def __init__(self, dreader, iminfo, im, seg_params):
    """
    See superclass Crypt_Finder
    """
    Crypt_Finder.__init__(self, dreader, iminfo, im, seg_params)

    if 'paneth' in dreader.seginfo:
      self.has_paneth = True
    else:
      self.has_paneth = False

    self.crypt_objects = dreader.crypt_objects_impath.readim(iminfo)
    self.dna_objects = dreader.dna_objects_impath.readim(iminfo)

    if self.has_paneth:
      self.paneth_objects = dreader.paneth_objects_impath.readim(iminfo)

    self.crypt_nuclei = []

    # output files
    self.seg_outpath = self.dreader.stem_objects_impath.get_path(self.iminfo)
    self.overlay_outpath = self.dreader.stem_overlay_impath.get_path(self.iminfo)

  def filter_paneth(self):
    """
    Filter out Paneth nuclei (assigned as nuclei closest to centroid of Paneth objects)
    """

    paneth_labels = imfuns.assign_centroids(self.im_segmented, self.paneth_objects)

    self.im_segmented = imfuns.remove_regions(self.im_segmented, paneth_labels)


  def filter_partial(self):
    """
    Filter out nuclei partially in the crypt. Partial nuclei are defined as nuclei
    where ratio of the area outside the crypt to the area inside the crypt > PARTIAL_RATIO
    """

    in_nuclei, out_nuclei, self.im_segmented = imfuns.overlap_regions(self.dna_objects, 
      self.crypt_objects, self.C['PARTIAL_RATIO'], extra_return=True)

    return in_nuclei, out_nuclei

  def preprocess(self):
    pass

  def segment(self):
    """
    Identify stem nuclei in crypts (filter out Paneth and partial nuclei)
    """

    self.filter_partial()
    
    if self.has_paneth:
      self.filter_paneth()
    
