"""
Segmentation and feature extraction
"""

from collections import OrderedDict
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
import logging

from datareader import ImageInfo
from utils import helperfuns, imfuns, segmentation, setting

class PlateSegmentation:
  """
  Handles segmentation for a plate

  Attributes:
    default_seg_params(dict): default segmentation parameters
    dreader (DataReader): contains experiment info
    object_lst (list): list of object_types in current plate
    seg_params (dict): plate specific segmentation parameters
    logger (Logger): logging object
  """

  def __init__(self, dreader, seg_params_path):
    """
    Loads attributes

    Args:
      dreader (DataReader): for current plate
      seg_params_path (str): path to seg_params file
    """

    self.dreader = dreader

    # load segmentation parameter file
    with open(seg_params_path) as f:
      self.seg_params = yaml.safe_load(f.read())

    self.default_seg_params = setting.DEFAULT_SEG_PARAMS

    self.object_lst = self.dreader.seginfo.keys()

    self.logger = logging.getLogger()

  def get_seg_params(self, object_type):
    """
    If plate parameters not present, fall back to default params

    Args:
      object_type (str): object to segment (e.g. dna, edu, crypt, muc2)

    Returns:
      dict: segmentation parameter for object type
    """

    try:
      seg_params = self.seg_params[object_type]
    except KeyError:
      seg_params = self.default_seg_params[object_type]

    return seg_params

  def run(self, object_type, plot=False): 
    """
    Runs segmentation on plate
    
    Args:
      object_type (str): object to segment (e.g. dna, edu, crypt, muc2)
    """
    
    seg_params = self.get_seg_params(object_type)

    # self.logger.notset(object_type, '\n', seg_params)

    for well in self.dreader.plateinfo.wells:
      wp = WellSegmentation(self.dreader, well, seg_params, object_type)
      wp.run(plot=plot)

  def run_all(self):
    """
    Runs all object segmentation on plate
    """
    for object_type in self.object_lst:
      self.run(object_type)

class WellSegmentation:
  """
  Handles segmentation for a well

  Attributes:
    ch_lst (list of str): list of channels used for segmentation
    dreader (DataReader): contains experiment info
    fields_iminfo (list of ImageInfo): all fields in well
    seg_params (dict): plate specific segmentation parameters
    segmentor (Segmentor): segmentation object
    well (Well): current well
    z_lst (list of int): list of z planes used for segmentation
  """

  def __init__(self, dreader, well, seg_params, object_type):
    """
    Loads attributes

    Args:
      dreader (DataReader): for current plate
      well (Well): well being segmented
      seg_params (dict): segmentation prameters
      object_type (str): object to segment (e.g. dna, edu, crypt, muc2)
    """
    segmentor_dic = {
      'dna': segmentation.Nuclear_Segmentor,
      'crypt': segmentation.Crypt_Finder,
      'goblet': segmentation.Goblet_Segmentor, 
      'edu': segmentation.EdU_Segmentor, 
      'singlelgr5': segmentation.Single_Lgr5_Segmentor, 
      'paneth': segmentation.Paneth_Segmentor,
      'ee': segmentation.EE_Segmentor,
      'stem': segmentation.Stem_Segmentor
    }

    self.dreader = dreader
    self.well = well
    self.seg_params = seg_params
    self.object_type = object_type

    self.fields_iminfo = self.dreader.plateinfo.field_iterator(wells=[well])

    seginfo = self.dreader.seginfo[object_type]
    self.segmentor = segmentor_dic[object_type]

  def readim(self, iminfo):
    """
    For given field, get maximum projection of all images (across z, channels)
    used for segmentation of object type

    Args:
      iminfo (ImageInfo): field of interest

    Returns:
      ndarray (float): maximum projected image
    """

    return self.dreader.get_max_im(iminfo, self.object_type)

  def run(self, plot=False):
    for iminfo in self.fields_iminfo:
      im = self.readim(iminfo)
      seg = self.segmentor(self.dreader, iminfo, im, self.seg_params)
      seg.run(plot=plot)

class FeatureExtraction:
  """
  Handles feature extraction functions

  Attributes:
    dreader (DataReader): contains experiment info
    features (list):
    logger (Logger): logging object
    object_lst (list): list of object_types in current plate
    readim_dict (dict): object_type mapping to function that reads in segmented
      image of that object type
  """

  def __init__(self, dreader):
    """
    Loads attributes

    Args:
      dreader (DataReader): for current plate
    """
    self.dreader = dreader

    self.features = []

    self.object_lst = self.dreader.seginfo.keys()

    self.logger = logging.getLogger()


  def count_num(self, fld, object_type):
    """
    Counts number of objects in a well. In the base case, this involves adding the
    counts in all fields. If fields are dropped, those fields are ignored. If fields
    have a substitute count, the substitute count is used instead.

    Args:
      well (Well): well to tally 
      object_type (str): object type to count

    Returns:
      int: number of objects of object_type in well
    """

    if isinstance(fld, ImageInfo):
      im = self.dreader.objects_impath[object_type].readim(fld)
      return imfuns.count_objects(im)
    elif isinstance(fld, int):
      return fld

  def more_feat_fn(self, feat_type):
    feat_fn = {
      'stemedu_num': self.count_stem_edu # edu+ stem cells
      }

    return feat_fn[feat_type]

  def count_stem_edu(self, fld):
    stem = self.dreader.objects_impath['stem'].readim(fld)
    edu = self.dreader.objects_impath['edu'].readim(fld)

    double_pos = imfuns.mask_im(edu, stem)

    return imfuns.count_objects(double_pos)

  def field_iterate(self, well, fn, obj_type=None, **kws):
    if obj_type is None:
      fld_lst = self.dreader.plateinfo.field_iterator(wells=[well])
    else:
      fld_lst = self.dreader.plateinfo.dropfield_iterator(obj_type, wells=[well])

    if not fld_lst:
      return np.nan

    count = 0

    for fld in fld_lst:

      count += fn(fld, **kws)

    return count

  def well_extract(self, well, more_features=[]):
    """
    Extract all features for a well

    Args:
      well (Well): well to tally 

    Returns 
      dict: of features
    """

    treatment = self.dreader.plateinfo.get_well_treatment(well)
    feats = OrderedDict([('row', well.row), ('col', well.col), ('treatment', treatment)])

    for object_type in self.object_lst:
      key = object_type+'_num'
      feats[key] = self.field_iterate(well, self.count_num, obj_type=object_type, object_type=object_type)

    for feat in more_features:
      feats[feat] = self.field_iterate(well, self.more_feat_fn(feat))
      
    return feats

  def save(self, feat_list):
    """
    Save features to file
    """
    path = self.dreader.feat_impath.template
    self.features = pd.DataFrame(feat_list)
    ordered_columns = list(feat_list[0].keys())
    self.features[ordered_columns].to_csv(path, index=False)
    
  def run(self, more_features=[]):

    feat_list = []

    for well in self.dreader.plateinfo.wells:
      feats = self.well_extract(well, more_features=more_features)
      feat_list.append(feats)
    self.save(feat_list)


