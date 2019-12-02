"""
Helpers image processing functions
==================================

"""
import numpy as np
from skimage import measure

max_pix = {
  'uint8': 255,
  'uint12': 4095,
  'uint16': 65535,
  'float64': 1
}

def assign_centroids(im_labeled, target_objects):
  """
  Assigned objects in im_labeled closest to centroid target objects)
  """

  centroids = [np.round(el.centroid).astype(int) for el in measure.regionprops(target_objects)]

  if not centroids:
    return [] 

  x, y = zip(*centroids)
  z = np.ones(len(x));

  target_seeds = np.zeros(target_objects.shape).astype(int)
  target_seeds[x,y] = z

  target_seeds_labeled = mask_im(im_labeled, target_seeds)
  target_labels = nonzero_list(target_seeds_labeled)

  return target_labels

def blend_ims(im_lst):
  downscale = 0.8
  composite_im = downscale * np.sum(im_lst, axis=0)
  composite_im[composite_im>(max_pix['uint8'])] = max_pix['uint8']
  return np.uint8(composite_im)

def convert_to_double(im, divideby=1):
  return im.astype(float)/divideby

def convert_to_uint8(im, datatype=None):

  if datatype is None:
    datatype = str(im.dtype)

  try:
    max_p = max_pix[datatype]
  except:
    print(datatype + ' is not a valid image type')

  return np.uint8((im / max_p)* max_pix['uint8'])

def count_objects(im):
  return len(nonzero_list(im))

def drop_zero(lst):
  """Returns list without 0 element"""

  nonzero = [el for el in lst if el != 0]

  return nonzero


def keep_regions(im, labels, bg_val=0):
  """Keep regions with label in given list from labeled image"""

  new_im = im.copy()
  new_im[~np.isin(new_im, labels)] = bg_val

  return new_im

def im_linscale(im_input, thresh_up, thresh_lo):

    if thresh_lo > thresh_up:
      raise ValueError('Lower threshold should be less than higher threshold!')

    im = np.copy(im_input)

    im[im<thresh_lo] = thresh_lo
    im[im>thresh_up] = thresh_up
    im -= thresh_lo
    im = im/np.max(im)*max_pix['uint8']
    return np.uint8(im)

def imthresh(im, thresh):
  """Sets pixels in image below threshold value to 0"""
  
  thresh_im = im.copy()
  thresh_im[thresh_im < thresh] = 0
  return thresh_im

def mask_im(im, mask, val=0):
  """Sets pixels not in mask to 0 (or val) in image"""

  masked_im = im.copy() 
  masked_im[mask == 0] = val
  return masked_im

def max_int_image(im_list):

  im_max = np.amax(im_list, axis=0)

  return im_max

def multi_otsu(im):

  # calculate image histogram
  nbins = np.amax(im) - np.amin(im)
  hist = np.histogram(im, bins=nbins)
  N = sum(hist[0])
  hist = list(zip(list(hist[0]), list(hist[1])))

  # mean int of image
  m = np.mean(im)

  # determined threshold
  thresh0 = 0
  thresh1 = 0

  # between class variance
  max_sigma = 0

  # weight and mean of first class
  w0_k = 0
  m0_k = 0

  # iterate through all threshold pairs
  for f0, t0 in hist:

    w0_k += f0 / N
    m0_k += t0 * (f0 / N)
    m0 = m0_k / w0_k

    w1_k = 0
    m1_k = 0

    for f1, t1 in [(x[0], x[1]) for x in hist if x[1] > t0]:
      w1_k += f1 / N
      m1_k += t1 * (f1 / N)
      m1 = m1_k / w1_k

      w2_k = 1 - (w0_k + w1_k)
      m2_k = m - (m0_k + m1_k)

      if w2_k > 0:    

        m2 = m2_k / w2_k

        curr_sigma = w0_k * (m0 - m)**2 + w1_k * (m1 - m)**2 + w2_k * (m2 - m)**2


        if max_sigma < curr_sigma:
          max_sigma = curr_sigma
          thresh0 = t0
          thresh1 = t1

  return (thresh0, thresh1)

def nonzero_list(im):
  """Returns image pixels as a list of unique labels without the 0s"""
  uniques = list(np.unique(im))
  nonzero = drop_zero(uniques)
  return nonzero

def overlap_regions(im, mask, partial_ratio, extra_return=False):
  """
  Filter out objects partially in the mask. Partial objects are defined as objects
  where ratio of the area outside the mask to the area inside the mask > partial_ratio
  """

  im_in = mask_im(im, mask)

  in_rp = measure.regionprops(im_in)
  in_areas = [el.area for el in in_rp]
  in_labels = [el.label for el in in_rp]

  im_potential = keep_regions(im, in_labels)
  im_out = mask_im(im_potential, ~(mask>0))

  out_rp = measure.regionprops(im_out)
  out_areas = [el.area for el in out_rp]
  out_labels = [el.label for el in out_rp]

  remove_labels = []

  for out_idx, label in enumerate(out_labels):
    out_area = out_areas[out_idx]

    in_idx = in_labels.index(label)
    in_area = in_areas[in_idx]

    if out_area/in_area > partial_ratio:
      remove_labels.append(label)

  im_filtered = remove_regions(im_potential, remove_labels)

  if extra_return:
    return im_in, im_out, im_filtered
  else:
    return im_filtered

def remove_regions(im, labels, bg_val=0):
  """Remove regions with label in given list from labeled image"""

  new_im = im.copy()
  new_im[np.isin(new_im, labels)] = bg_val

  return new_im

def stitch_well(wellmap, im_dic, empty_fld=-1):

  # check that ims are all same size
  im_dim = None
  for im in im_dic.values():
    if im_dim == None:
      im_dim = im.shape
    else:
      if im_dim != im.shape:
        raise ValueError('All images must be of the same size')

  im_h = im_dim[0]
  im_w = im_dim[1]

  fld_h = wellmap.shape[0]
  fld_w = wellmap.shape[1]

  well_h = im_h*fld_h
  well_w = im_w*fld_w

  if len(im_dim) == 3:
    well = np.zeros((well_h, well_w, im_dim[2]))
  else:
    well = np.zeros((well_h, well_w))

  well = well.astype(im_dic[0].dtype)

  for row in range(fld_h):
    for col in range(fld_w):
      idx = wellmap[row][col]
      if idx != empty_fld:
        if len(im_dim) == 3:
          well[row*im_h : (row+1)*im_h, col*im_w : (col+1)*im_w, :] = im_dic[idx]
        else: 
          well[row*im_h : (row+1)*im_h, col*im_w : (col+1)*im_w] = im_dic[idx]

  return well

def subtract(im1, im2):
  """Subtract two images. Pixel value cannot go below 0"""

  im = im1 - im2
  im[im<0] = 0

  return im


