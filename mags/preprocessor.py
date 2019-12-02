"""
Handles data processing for analysis and visualization
"""

import cv2
import glob
import logging
import os
import re
import shutil
import subprocess
import time
import warnings

from datareader import DataReader, Well, WellInfo, ImageInfo
from utils import imfuns, setting

class Preprocessor:
  """
  Handles pre-processing steps for image analysis and visualization

  Attributes:
    dreader (DataReader): contains experiment info
    bfsetting (BFConvertSettings): input for bfconvert program
    nd2_wellpattern (regex str): regex to search for Well tags in nd2 files
    ext (str): output image file extension (usually tiff)
  """

  def __init__(self, dreader, silent=True):
    """
    Loads attributes

    Args:
      dreader (DataReader): for current plate 
    """

    self.dreader = dreader

    self.bfsetting = setting.BFConvertSettings(timelapse=dreader.plateinfo.is_timelapse)
    self.nd2_wellpattern = '(?<={tag:s}).{{3}}'.format(tag=self.dreader.well_tag)
    self.ext = dreader.get_file_extension()

    self.logger = logging.getLogger()

    if silent:
      self.pipe = subprocess.DEVNULL
    else:
      self.pipe = None

  def build_pyr(self, wells=None, pyrloop=1):
    """
    Create downsized images ("pyramid") 

    Args: 
      wells (list of Wells): wells to build pyramid images, if None, will use all wells
      pyrloop (int): number of times to downsize (1 --> 1/2 in all dimensions)

    Returns: None
    """

    iminfo_stack = self.dreader.plateinfo.image_iterator(wells=wells)

    for iminfo in iminfo_stack:

      # read image
      im = self.dreader.data_impath.readim(iminfo)

      im_small = self.downsize_image(im, pyrloop)

      self.dreader.pyr_impath.saveim(iminfo, im_small) 

  def convert_nd2(self):
    """
    Converts nd2 files to tiff files organized by folders

    Args: None

    Returns: None
    """

    nd2_files = self.dreader.get_nd2_filelist()

    self.logger.info(nd2_files)

    if len(nd2_files) == 0:
      warnings.warn('No nd2 file found in microscope path')

    for nd2_f in nd2_files:

      outpath = self.create_well_folder(nd2_f)

      # run bfconvert
      self.run_bfconvert(nd2_f, outpath)

      self.rename_tiff_files(os.path.dirname(outpath))

      if not self.resave_tiffs(os.path.dirname(outpath)):
        raise ValueError('Convert failed for '+nd2_f)

  def create_well_folder(self, nd2_file):
    """
    Create well folder for converted tiffs
  
    Args:
      nd2_file (str): path to nd2 file

    Returns:
      str: path template for files in well folder

    """
    match = re.search(self.nd2_wellpattern, nd2_file)
    if match is not None:
      w = Well(row=match.group()[0], col=match.group()[1:])

      outdir = os.path.dirname(self.dreader.data_impath.template).format(row=w.row, col=w.col)
      if not os.path.exists(outdir):
        os.makedirs(outdir)
      return os.path.join(outdir, self.dreader.name+self.bfsetting.format)
    else:
      warnings.warn('File not used. Well tag not found: '+nd2_file)


  def downsize_image(self, im, pyrloop):
    """
    Down size image a given number of times (each time reduce size by 1/4)

    Args: 
      im (ndarray): original image
      pyrloop (int): number of times to reduce

    Returns: 
      ndarray (uint8): downsized image #UPDATE
    """

    # im = imfuns.convert_to_uint8(im, datatype=self.dreader.im_dtype)

    for i in range(pyrloop):
      im = cv2.pyrDown(im)

    return im

  def rename_tiff_files(self, outdir):
    """
    Rename tiffs created by bftools to preset filename format

    Args:
      outdir (str): path to well directory containing tiff files (e.g '/.../B11/')
    
    Returns: None
    """

    tiff_files = glob.glob(os.path.join(outdir, '*'+self.ext))

    for tiff_f in tiff_files:
      match_info = re.search(self.bfsetting.fname_pattern, tiff_f)

      row = match_info.group('row')
      col = int(match_info.group('col'))
      fld = int(match_info.group('fld'))
      z = int(match_info.group('Z'))
      ch = match_info.group('ch')

      iminfo = ImageInfo(row=row, col=col, z=z, fld=fld, ch=ch)
      
      new_path = self.dreader.data_impath.get_path(iminfo)

      shutil.move(tiff_f, new_path)

  def resave_tiffs(self, outdir):
    """ 
    Re-save images using FIJI (need to do this for some reason..)

    Args:
      outdir (str): path to well directory containing tiff files (e.g '/.../B11/')

    Returns:
      bool: True if successful
    """
    fiji_path = setting.FIJI['path']
    macro_path = setting.FIJI['batch_tiff_save_macro']

    filelist = glob.glob(os.path.join(outdir, '*'+self.ext))

    subprocess.run(fiji_path+' --headless --system -macro '+macro_path+' '+outdir, shell=True,
      stdout=self.pipe, stderr=self.pipe)

    time.sleep(120)
    time_cap = 20*60 # 20min
    end_time = time.time() + time_cap

    while time.time() < end_time:
      if all([cv2.imread(path, cv2.IMREAD_ANYDEPTH) is not None for path in filelist]):
        return True
      else:
        time.sleep(5)

    return False

  def run_bfconvert(self, nd2_file, outpath):
    subprocess.run(self.bfsetting.path+' '+nd2_file+' '+outpath, shell=True,
      stdout=self.pipe, stderr=self.pipe)

  def stitch_well(self, wellinfo):
    """ 
    Stitches fields in a well to create a whole well image

    Args:
      wellinfo (WellInfo): well to create image (include z, ch info)

    Returns: 
      ndarray: stitched image
    """

    im_dic = {}

    for fld in self.dreader.plateinfo.field_lst:

      iminfo = ImageInfo(row=wellinfo.row, col=wellinfo.col, z=wellinfo.z, fld=fld, ch=wellinfo.ch)

      # read image
      im = self.dreader.pyr_impath.readim(iminfo)
      im_dic[fld]  = im

    # stitch well
    well_im = imfuns.stitch_well(self.dreader.plateinfo.fieldmap, im_dic)

    return well_im


  def make_previews(self, wells=None):
    """ 
    Stitches fields in a well to create a whole well image

    Args:
      wells (list of Wells): wells to create the image. If None, use all wells

    Returns: None
    """

    wellinfos = self.dreader.plateinfo.wellinfo_iterator(wells=wells)

    for wellinfo in wellinfos:
          
      well_im = self.stitch_well(wellinfo)

      self.dreader.wellpreview_impath.saveim(wellinfo, well_im)
      
