"""
Stores info about data, reads images etc
"""

import cv2
import glob
import logging
import numpy as np
import os
import pandas as pd
import random
import re
from skimage import io
import warnings
import yaml
import logging

from utils import helperfuns, imfuns, setting

class DataReader:
  """
  Plate info encapsulation (paths, plate setup, etc). Mostly handles paths definition.

  Attributes:
    config_path (str): path to config file that creates the object
    conf (dict): all plate config information
    dropfield_df (DataFrame): dropfield info if exists
    im_dtype (str): raw data bit info (e.g. uint12)
    is_well_nd2_files (bool): True if nd2 files are divided by wells
    max_px_val (float): maximum value stored (e.g. 4095 for uint12)
    plateinfo (PlateInfo): plate setup info
    seginfo (dict): segmentation settings
    viz_settings (dict): visualization settings
    well_tag (str): precedes well info in nd2 filenames
  """

  def __init__(self, config_path):
    """
    Loads config file, plate info, paths, and visualization and segmentation settings

    Args:
      config_path (str): path to configuration file for plate 
    """

    self.config_path = config_path

    # load config file
    with open(config_path) as f:
      self.conf = yaml.safe_load(f.read())

    self.exp_name = self.conf['exp_name']
    self.plate_num = self.conf['plate_num']
    self.name = self.conf['plate_name']

    dropfield_df = self.load_dropfile()

    self.seginfo = self.conf['seg_info']

    self.plateinfo = PlateInfo(self.conf, dropfield_df=dropfield_df)

    self.im_dtype = self.conf['data_type']
    self.max_px_val = setting.max_pixel_val_dic[self.im_dtype]
    self.well_tag = self.conf['well_tag']
    self.is_well_nd2_files = self.conf['well_nd2_files']

    self.setup_paths()

    self.segdrop_df = self.load_segdropfile()

    self.viz_settings = self.conf['viz_settings']

  def default_impath(self, dir, fname):
    return ImagePath(dir, fname, path_prefix=self.conf['path_prefix'], fname_prefix=self.conf['fname_prefix'])

  def get_max_im(self, iminfo, object_type, z_lst=None, marker=None, data_source='raw'):
    """Max projection of image using the planes specified for segmentation (default)"""

    if data_source == 'raw':
      readim = self.data_impath.readim
    elif data_source == 'well':
      readim = self.wellpreview_impath.readim

    if z_lst is None:
      z_lst = self.seginfo[object_type]['zplanes']

    if marker is None:
      ch_lst = self.plateinfo.markers[self.seginfo[object_type]['marker']]
    else:
      ch_lst = self.plateinfo.markers[marker]

    iminfo_stack = self.plateinfo.get_iminfo_stack(iminfo, z_lst=z_lst, ch_lst=ch_lst)

    ims = [readim(info) for info in iminfo_stack]
    im_max = imfuns.max_int_image(ims)

    return imfuns.convert_to_double(im_max, divideby=self.max_px_val)

  def get_nd2_filelist(self):
    return glob.glob(os.path.join(self.microscope_path, '*.nd2'))

  def get_overlay_im(self, iminfo, object_type):
    return self.overlay_impath[object_type].readim(iminfo)

  def get_file_extension(self):
    return self.conf['data_filename'].split('.')[-1]

  def get_objects_im(self, iminfo, object_type):
    return self.objects_impath[object_type].readim(iminfo)

  def load_dropfile(self, fname=None):
    """Loads dropfile into a dataframe if file exists and is marked complete"""

    if fname is None:
      fname = setting.DROPFILE['filename']

    dropfield_path = os.path.join(os.path.dirname(self.config_path), fname)

    dropfield_df = None

    try:
      df = pd.read_csv(dropfield_path)

      # check for 'done' in columns, marking QC as completed
      if setting.DROPFILE['done'] not in df.columns.values:
        warnings.warn('Drop file not completed. Drop file not used. ({:s})'.format(self.exp))

      # check no duplicate rows
      elif any(df.duplicated(subset=['row', 'col', 'fld'])):
        warnings.warn('Drop file has duplicated row/col/fld values. Drop file not used. ({:s})'.format(self.exp))
      else:
        dropfield_df = df

    except:
      warnings.warn('Drop file not found.')

    return dropfield_df

  def load_segdropfile(self):

    if os.path.exists(self.segdrop_path):
      df = pd.read_csv(self.segdrop_path)

    else:
      df = None

    return df

  def setup_paths(self):
    """Creates and stores path objects"""
    imp = setting.IMPATH

    self.microscope_path = self.conf['microscope_path']

    self.segdrop_path = os.path.join(os.path.dirname(self.config_path), setting.SEGDROPFILE['filename'])

    self.data_impath = self.default_impath(self.conf['data_path'], self.conf['data_filename']) 

    self.pyr_impath = self.default_impath(imp['pyr_path'], imp['pyr_filename'])
    self.wellpreview_impath = self.default_impath(imp['well_preview_path'], imp['well_preview_filename'])

    self.dna_objects_impath = self.default_impath(imp['seg_path'], imp['dna_objects_filename'])
    self.dna_overlay_impath = self.default_impath(imp['seg_path'], imp['dna_overlay_filename'])

    self.crypt_objects_impath = self.default_impath(imp['seg_path'], imp['crypt_objects_filename'])
    self.crypt_overlay_impath = self.default_impath(imp['seg_path'], imp['crypt_overlay_filename'])

    self.goblet_objects_impath = self.default_impath(imp['seg_path'], imp['goblet_objects_filename'])
    self.goblet_overlay_impath = self.default_impath(imp['seg_path'], imp['goblet_overlay_filename'])

    self.edu_objects_impath = self.default_impath(imp['seg_path'], imp['edu_objects_filename'])
    self.edu_overlay_impath = self.default_impath(imp['seg_path'], imp['edu_overlay_filename'])

    self.singlelgr5_objects_impath = self.default_impath(imp['seg_path'], imp['singlelgr5_objects_filename'])
    self.singlelgr5_overlay_impath = self.default_impath(imp['seg_path'], imp['singlelgr5_overlay_filename'])

    self.paneth_objects_impath = self.default_impath(imp['seg_path'], imp['paneth_objects_filename'])
    self.paneth_overlay_impath = self.default_impath(imp['seg_path'], imp['paneth_overlay_filename'])

    self.ee_objects_impath = self.default_impath(imp['seg_path'], imp['ee_objects_filename'])
    self.ee_overlay_impath = self.default_impath(imp['seg_path'], imp['ee_overlay_filename'])

    self.stem_objects_impath = self.default_impath(imp['seg_path'], imp['stem_objects_filename'])
    self.stem_overlay_impath = self.default_impath(imp['seg_path'], imp['stem_overlay_filename'])

    self.objects_impath = {
      'dna': self.dna_objects_impath,
      'crypt': self.crypt_objects_impath,
      'edu': self.edu_objects_impath,
      'ee': self.ee_objects_impath,
      'goblet': self.goblet_objects_impath,
      'paneth': self.paneth_objects_impath,
      'singlelgr5': self.singlelgr5_objects_impath,
      'stem': self.stem_objects_impath
      }

    self.overlay_impath = {
      'dna': self.dna_overlay_impath,
      'crypt': self.crypt_overlay_impath,
      'edu': self.edu_overlay_impath,
      'ee': self.ee_overlay_impath,
      'goblet': self.goblet_overlay_impath,
      'paneth': self.paneth_overlay_impath,
      'singlelgr5': self.singlelgr5_overlay_impath,
      'stem': self.stem_overlay_impath
      }

    self.feat_impath = self.default_impath(imp['feat_path'], imp['feat_filename'])

    self.output_path = os.path.join(self.conf['path_prefix'], self.conf['output_path'])

    self.segviz_impath = self.default_impath(self.conf['output_path'], imp['segviz_filename'])

class ImagePath:
  """
  Path object stores path string and read/write image data to path. Typically as a string format
  
  Attributes:
    dir (str): directory of files
    fname (str): filename template
    keys (str): string format keys in template
    template (str): full path template
  """

  def __init__(self, folder, fname, path_prefix='', fname_prefix=''):
    """Path constructed as path_prefix/dir/fname_prefix+fname"""

    self.folder = os.path.join(path_prefix, folder)
    self.fname = fname_prefix+fname
    self.template = os.path.join(self.folder, self.fname)
    self.keys = re.findall(r"(?:[^{]|^){(\w+):", self.template)

  def get_path(self, iminfo):
    """Pulls out relevant keys and formats path"""

    if iminfo is None:
      return self.template
    else:
      fname_subs = {k: v for k, v in iminfo.__dict__.items() if k in self.keys}

      if set(fname_subs.keys()) != set(self.keys):
        raise ValueError('Must provide all keys')
        
      return self.template.format(**fname_subs)

  def readim(self, iminfo):
    path = self.get_path(iminfo)
    im = io.imread(path)
    # im = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

    if im is None:
      raise ValueError('Cannot read image at '+path)
    else:
      return im

  def saveim(self, iminfo, im):
    path = self.get_path(iminfo)
    return cv2.imwrite(path, im)
    
class Well:
  """
  Simple well object
  
  Attributes:
    row (str): row position of well (upper case)
    col (int): column position of well 
  """

  def __init__(self, row=None, col=None):
    if row is None or col is None:
      raise ValueError('Must provide row and column of well')

    self.row = row.upper()
    self.col = int(col)

  def isin(self, well_list):
    """Return True if is in list of wells"""

    return any([self.sameas(w) for w in well_list])

  def printinfo(self):
    return('row: {:s}, col: {:02d}'.format(self.row, self.col))

  def sameas(self, well):
    """Return True if same as given well"""

    if self.row == well.row and self.col == well.col:
      return True
    else:
      return False

class WellInfo(Well):
  """
  Well object with additional information of the image
  
  Attributes:
    ch (str): channel
    t (int): time
    z (int): z position 
  """

  def __init__(self, row=None, col=None, z=None, ch=None, t=None):
    Well.__init__(self, row=row, col=col)
    self.z = None if z is None else int(z)
    self.ch = None if ch is None else str(ch)
    self.t = None if t is None else float(t)

  def printinfo(self):
    z = 'None' if self.z is None else str(self.z)
    ch = 'None' if self.ch is None else str(self.ch)
    t = 'None' if self.t is None else str(self.t)

    return(Well.printinfo(self) + ', z: {:s}, ch: {:s}, t: {:s}'.format(z, ch, t))

  def sameas(self, w):
    """Return True if same as given well info"""

    if Well.sameas(self, w) and self.z==w.z and self.ch==w.ch and self.t==w.t:
      return True
    else:
      return False

class ImageInfo(WellInfo):
  """
  WellInfo object with specification about the field of view
  
  Attributes:
    fld (int): field number
  """
  def __init__(self, row=None, col=None, fld=None, z=None, ch=None, t=None):

    if fld is None:
      raise ValueError('Must provide field index of image')
      
    WellInfo.__init__(self, row=row, col=col, z=z, ch=ch, t=t)
    self.fld = int(fld)

  def printinfo(self):
    return (WellInfo.printinfo(self) + ', fld: {:02d}'.format(self.fld))

  def sameas(self, iminfo):
    """Return True if same as given well"""

    if WellInfo.sameas(self, iminfo) and self.fld == iminfo.fld:
      return True
    else:
      return False

class PlateInfo:  
  """
  Plate structure info (wells used, field map, etc), get certain wells/fields, get iterator over wells/fields
  
  Attributes:
    channels (list str): imaging channels ('C1', 'C2', ...)
    dropfield_df (DataFrame): drop field info
    field_lst (list int): list of field numbers
    fieldmap (array): field arrangment in a grid
    is_timelapse (bool): True if timelapse data, False otherwise
    markers (dict of (str, list)): marker name and channels for that marker
    name (str): plate identifier
    platemap (array): indicates used wells and conditions in used wells
    platetype (str): size of plate (e.g. '96_well')
    t_lst (list int): list of time points in experiment
    wells (list Well): list of used wells as Well objects
    z_lst (list int): list of z stack positions
  """
  def __init__(self, conf, dropfield_df=None):

    self.platemap = np.array(conf['plate_map'])
    self.platetype = self.check_platetype()

    self.wells = self.platemap_to_wells()

    self.fieldmap = np.array(conf['field_map'])
    self.field_lst = helperfuns.convert_to_sequence(conf['field_start'], conf['field_num'])

    self.channels = list(conf['channels'].keys()) # channel_lst -> channels
    self.markers = {}    # channels -> markers

    # a marker may be imaged in multipled "channels" if multiple z stack groups are used
    for ch, marker in conf['channels'].items():
      self.markers.setdefault(marker,[]).append(ch) 

    self.z_lst = helperfuns.convert_to_sequence(conf['z_start'], conf['z_num'])

    self.is_timelapse = conf['t_start'] is not None
    if self.is_timelapse:
      self.t_lst = helperfuns.convert_to_sequence(conf['t_start'], conf['t_num'])
    else:
      self.t_lst = [None]

    self.dropfield_df = dropfield_df

    self.logger = logging.getLogger()

  def check_ch(self, ch):
    """Return True if ch is valid value. Input can be list or single value"""
    return self.check_input(ch, self.channels)

  def check_isdropfield(self, iminfo, object_type): # check_drop -> check_isdropfield
    """
    Returns True if field is marked to be dropped, False if it is not, 
    or an int if there is an alternate count.
    """
    
    if self.dropfield_df is not None:
      row = self.lookup_imtable(iminfo, self.dropfield_df)
      row = row[object_type].tolist()

      if len(row) > 0:
        
        dropmark = row[0]

        # drop field
        if dropmark == 'x':
          return True

        # save field
        elif pd.isnull(dropmark):
          return False

        # substitute field
        elif helperfuns.check_numeric(dropmark):
          return int(dropmark)
        
        else:
          raise ValueError('Does not recognize dropmark', dropmark)
      
      else:
        # row is not marked to be dropped
        return False

    else:
      warnings.warn('No dropfield file found. No fields were dropped')
      return False
      
  def check_input(self, input, target):
    """Return True if input is in target. Input can be list or single value"""

    target = target + [None]

    if isinstance(input, list):
      return helperfuns.check_sublist(input, target)
    else:
      return input in target

  def check_marker(self, marker):
    """Return True if marker is valid value. Input can be list or single value"""
    return self.check_input(marker, list(self.markers.keys()))

  def check_platetype(self):
    """Look up plate type using column/row info from platemap"""

    platetype_lookup = {}
    for key, d in setting.PLATE_TEMPLATE.items():
      platetype_lookup[(d['num_rows'], d['num_cols'])] = key

    try: 
      return platetype_lookup[self.platemap.shape]
    except:
      raise ValueError('Plate map type not recognized')

  def check_well(self, well):
    """Return True if well is valid. Input is well object"""
    return well.isin(self.wells)

  def check_z(self, z):
    """Return True if z is valid value. Input can be list or single value"""
    return self.check_input(z, self.z_lst)

  def get_iminfo_stack(self, iminfo, z_lst=[None],  ch_lst=[None]):
    """
    Given an ImageInfo, generates a list of images of the same field but with 
    all combinations of z stack and channel values.

    Args:
      iminfo (ImageInfo): provides well, field, t information

    Returns: 
      list of ImageInfo: 
    """
    
    if z_lst == 'all':
      z_lst = self.z_lst

    if ch_lst == 'all':
      ch_lst = self.channels

    if not (self.check_z(z_lst) and self.check_ch(ch_lst)):
      raise ValueError('Inappropriate z and ch values')

    if type(z_lst) is list and type(ch_lst) is list:
      return [ImageInfo(row=iminfo.row, col=iminfo.col, t=iminfo.t, z=z, ch=ch, fld=iminfo.fld) for z in z_lst for ch in ch_lst]
    else:
      raise ValueError('Provide z and ch as a list')

  def get_plate_index(self):
    """
    Returns an array of same size as the plate with the following format:

    Well(row='A', col=1), Well(row='A', col=2), Well(row='A', col=3), ...
    Well(row='B', col=1), Well(row='B', col=2), Well(row='B', col=3), ...
    Well(row='C', col=1), Well(row='C', col=2), Well(row='C', col=3), ...
    ...
    ...
    ...
    """

    plate_info = setting.PLATE_TEMPLATE[self.platetype]

    rowmap = np.tile(np.transpose(np.array([plate_info['rows']])), (1, plate_info['num_cols']))
    colmap = np.tile(np.array(plate_info['cols']), (plate_info['num_rows'], 1))

    return np.array([[Well(row=w[0], col=w[1]) for w in x] for x in np.dstack((rowmap, colmap))])

  def get_random_well(self, n=1):
    """
    Generate random well(s) that are used in the plate. 

    Args:
      n (int, default=1): number of random wells

    Returns:
      Well object: if n=1
      list of Well objects: if n > 1
    """

    if n == 1:
      return np.random.choice(self.wells, size=n, replace=False)[0]
    elif n > 1 and n <= len(self.wells):
      return np.random.choice(self.wells, size=n, replace=False)
    else:
      raise ValueError('Please provide positive integer for well number')

  def get_random_field(self, z=None):
    """
    Generate a random field that are used in the plate. Note that this is concerned
    with the position (row, col, fld), not image properties (z, ch, t)

    Args:
      z (int, default=None): z stack

    Returns:
      ImageInfo object: if z is not a list 
      list of ImageInfo objects: otherwise
    """
    well = self.get_random_well()
    fld = np.random.choice(self.field_lst)

    if not self.check_z(z):
      raise ValueError('Invalid z value')

    if type(z) is list:
      return [ImageInfo(row=well.row, col=well.col, z=i, fld=fld) for i in z]
    else:
      return ImageInfo(row=well.row, col=well.col, z=z, fld=fld)

  def get_random_image(self):
    """
    Generate random images(s) that are in the dataset. Randomly selects valid values 
    for row, col, fld, z, ch, t.

    Args: None

    Returns: ImageInfo object
    """
    well = self.get_random_well()
    return ImageInfo(row=well.row, col=well.col, t=random.choice(self.t_lst), z=random.choice(self.z_lst), 
      ch=random.choice(self.channels), fld=random.choice(self.field_lst))

  def get_well_treatment(self, well):
    """Return treatment condition for well"""
    return self.platemap[helperfuns.well_to_coord(well, self.platetype)]

  def field_iterator(self, wells=None, ch=None): 
    """
    Generate list of all fields in a given set of wells

    Args:
      wells (list of Wells): if none given, will use all wells in plate
      ch (str): specify channel, otherwise None

    Returns:
      list of ImageInfos: where ch is None or a given value, and z, t are None values
    """

    if wells is None:
      wells = self.wells

    return [ImageInfo(row=w.row, col=w.col, fld=fld, z=None, ch=ch, t=None) for w in wells 
      for fld in self.field_lst]

  def dropfield_iterator(self, object_type, wells=None, ch=None):
    """
    Generate list of all not-dropped fields in a given set of wells

    Args:
      object_type: object checked for drop condition (e.g. goblet)
      wells (list of Wells): if none given, will use all wells in plate
      ch (str): specify channel, otherwise None

    Returns:
      list of ImageInfos/ints: See 'field_iterator'. Ints in places where substitutions are made
    """

    if wells is None:
      wells = self.wells

    all_flds = self.field_iterator(wells=wells, ch=ch)

    new_flds = []

    # for logging
    nochange = True
    header = 'Dropped/Changed fields for {:s}:'.format(object_type)

    for f in all_flds:
      dropfield = self.check_isdropfield(f, object_type)
      self.logger.info(dropfield)
      if dropfield is True:
        if nochange:
          nochange = False
          self.logger.info(header)

        self.logger.info('Well {row:s}{col:02d}, XY {fld:02d}'.format(row=f.row, col=f.col, fld=f.fld))
      elif dropfield is False:
        new_flds.append(f)

      elif isinstance(dropfield, int):
        new_flds.append(dropfield)

        if nochange:
          nochange = False
          self.logger.info(header)

        self.logger.info('Well {row:s}{col:02d}, XY {fld:02d} -> {num:d}'.format(row=f.row, col=f.col, fld=f.fld, num=dropfield))
      else:
        raise ValueError('Unrecognized input '+dropfield)

    return new_flds

  def image_iterator(self, wells=None):
    """
    Generate list of all images in a given set of wells

    Args:
      wells (list of Wells): if none given, will use all wells in plate

    Returns:
      list of ImageInfos: containing all possible fld, z, ch, t for given wells
    """

    if wells is None:
      wells = self.wells

    return [ImageInfo(row=w.row, col=w.col, fld=fld, z=z, ch=ch, t=t) for w in wells 
      for fld in self.field_lst for z in self.z_lst for t in self.t_lst for ch in self.channels] 

  def wellinfo_iterator(self, wells=None):
    """
    Generate list of wellinfo in a given set of wells

    Args:
      wells (list of Wells): if none given, will use all wells in plate

    Returns:
      list of WellInfo: containing all possible z, ch for given wells
    """

    if wells is None:
      wells = self.wells

    return [WellInfo(row=w.row, col=w.col, z=z, ch=ch) for w in wells 
      for z in self.z_lst for ch in self.channels] 

  def lookup_imtable(self, iminfo, df, object_type=None):
    """Find row in DataFrame based on ImageInfo"""

    iminfo_match = (df['row'] == iminfo.row) & (df['col'] == iminfo.col) & (df['fld'] == iminfo.fld)
    if object_type is None:
      return df[iminfo_match]
    else:
      return df[iminfo_match & (df['object_type'] == object_type)]


  def platemap_to_wells(self, platemap=None):
    """
    Converts plate map to a list of used wells

    Args:
      platemap(ndarray of bool): matrix containing True if well is used. If None, use current platemap

    Returns: 
      list of Well: wells that are used
    """

    if platemap is None:
      platemap = self.platemap.astype(bool)

    plate_index = self.get_plate_index()

    if platemap.shape != plate_index.shape:
      raise ValueError('Incorrect platemap format')

    return plate_index[platemap]

