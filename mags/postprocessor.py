"""
Post-processing steps such as removing incorrectly segmented objects
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set(rc={'image.cmap': u'Greys_r'})
from skimage import color

from datareader import ImageInfo
from utils import imfuns

class SegmentationCorrection:
  """
  Removes incorrect segmentation objects from segmentation masks

  Attributes:
    dreader (DataReader): contains experiment info
    segdrop_df (Dataframe): segmentation drop information
    iminfo_df (Dataframe): unique wells and object type combinations in segdrop dataframe
  """

  def __init__(self, dreader, printdrops=True):
    """
    Loads attributes

    Args:
      dreader (DataReader): for current plate
      printdrops (bool): prints objects dropped
    """
    self.dreader = dreader
    self.segdrop_df = self.dreader.segdrop_df

    self.iminfo_df = self.segdrop_df[['row', 'col', 'fld', 'object_type']].drop_duplicates()

    self.printdrops = printdrops

  def iminfo_from_row(self, row):
    return ImageInfo(row=row['row'], col=row['col'], fld=row['fld'])

  def check_dropdf(self):
    if self.segdrop_df is None:
      print('No segmentation drop file found')
      return False
    else:
      if self.printdrops:
        print('Dropping segmentation objects for {:s}'.format(self.dreader.name))
      return True

  def run(self):

    if not self.check_dropdf():
      return False

    for index, row in self.iminfo_df.iterrows():
      iminfo = self.iminfo_from_row(row)
      object_type = row['object_type']

      im_correction = ImageCorrection(self.dreader, iminfo, object_type)
      im_correction.run()
      im_correction.save()

      if self.printdrops:
        print(im_correction.print_dropinfo())

  def test_run(self):

    if not self.check_dropdf():
      return False

    for index, row in self.iminfo_df.iterrows():
      iminfo = self.iminfo_from_row(row)
      object_type = row['object_type']

      im_correction = ImageCorrection(self.dreader, iminfo, object_type)

      im_correction.run()
      im_correction.show_result()

      if self.printdrops:
        print(im_correction.print_dropinfo())

class ImageCorrection():
  """
  Removes incorrect segmentation objects for one image

  Attributes:
    dreader (DataReader): contains experiment info
  """

  def __init__(self, dreader, iminfo, object_type):
    """
    Loads attributes

    Args:
      dreader (DataReader): for current plate
      iminfo (ImageInfo): image to run
      object_type (str): type of object of the segmentation
    """
    self.dreader = dreader
    self.iminfo = iminfo
    self.object_type = object_type

    self.drop_ids = self.get_drop_ids()

    self.fig_w = 20
    self.fig_h = 10

  def create_mask(self,im):
    im = np.copy(im)

    im[im>0] = 1
    return im

  def get_drop_ids(self):
    drop_ids = self.dreader.plateinfo.lookup_imtable(self.iminfo, self.dreader.segdrop_df, 
      object_type=self.object_type)
    drop_ids = imfuns.drop_zero(drop_ids['drop_seg_id'].tolist())

    return drop_ids

  def drop_objects(self):
    self.old_seg = self.dreader.get_objects_im(self.iminfo, self.object_type)
    self.new_seg = imfuns.remove_regions(self.old_seg, self.drop_ids)

  def print_dropinfo(self):
    return '{:s} | {:s} dropped: {:s}'.format(self.iminfo.printinfo(), 
      self.object_type, ', '.join([str(el) for el in self.drop_ids]))

  def rename_old_path(self, path):
    filename, ext = os.path.splitext(path)
    new_path = filename+'_old'+ext

    return new_path

  def run(self):
    self.drop_objects()

  def save(self):

    impath = self.dreader.objects_impath[self.object_type]

    # save a copy of old segmentation
    oldpath = impath.get_path(self.iminfo)
    newpath = self.rename_old_path(oldpath)
    os.rename(oldpath, newpath)

    impath.saveim(self.iminfo, self.new_seg)

  def show_result(self):
    im = self.dreader.get_max_im(self.iminfo, self.object_type)

    fig, ax = plt.subplots(1,2, figsize=(self.fig_w, self.fig_h), sharex='all', sharey='all')

    # merge seg such that 2 means kept regions, 1 means removed regions, and 0 is non-regions
    merged_seg = self.create_mask(self.new_seg) + self.create_mask(self.old_seg)

    ax[0].imshow(im, zorder=1)
    ax[0].set_title('Raw Image')
    ax[1].imshow(color.label2rgb(merged_seg, bg_label=0), zorder=1)
    ax[1].set_title('Segmentation')

    plt.suptitle(self.print_dropinfo())

    plt.show()




      