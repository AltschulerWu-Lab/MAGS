"""
Other helper functions
"""

import contextlib
import fnmatch
import glob
import io
import logging
import logging.config
import numpy as np
import os
import re
import shutil
import sys
import time
import warnings
import yaml

# load constants
with open('config/constants.yaml') as f:
  constants = yaml.safe_load(f.read())

def check_numeric(x):
  """Return True if x is numeric. x can be float, int, or str of int ('5' not '5.0')"""

  if isinstance(x, int) or isinstance(x, float):
    return True

  if isinstance(x, str) and x.isnumeric():
    return True
  else:
    return False

def check_sublist(input_list, target_list):
  """Return True if every element in input is in target"""

  for el in input_list:
    if el not in target_list:
      return False
  return True

def convert_to_sequence(start, num, inc=1):
  """Given start number, size of list, and step size: return list"""
  return list(range(start, start + inc*num, inc))

def get_config_path(prefix, subdir='', local=False):
  """Return path to config file given plate folder, assuming default setup"""

  if local:
    fname = 'datainfo_local.yml'
  else:
    fname = 'datainfo.yml'

  return os.path.join(prefix, subdir, 'config', fname)

def get_cols(platetype):
  """Get list of columns for platetype"""
  return constants['plate_info'][platetype]['cols']

def get_rows(platetype):
  """Get list of rows for platetype"""
  return constants['plate_info'][platetype]['rows']

def rename_data_files(dir, pattern, old_str, new_str, ext='tiff', test=False):
  """
  Rename all files recursively in a directory with a given file extension

  Args:
    dir     (str): path to directory
    pattern (str): pattern to match in files
    old_str (str): substring to be replaced
    new_str (str): substring being put in 
    ext     (str): file extension
    test    (bool): testing mode - if true, replacements are printed but not made

  Returns:
    bool: True if files found and matches made. False otherwise
  """

  filelst = glob.glob(os.path.join(dir, '*'+ext))

  found = False

  for f in filelst:
    search = re.search(pattern, f)
    
    if search is not None:
      match = search.group()
      repl_match = str.replace(match, old_str, new_str)
      new_f = str.replace(f, match, repl_match)
      if test:
        print(f)
        print(new_f)
      else:
        shutil.move(f, new_f)

      found = True

  return found

def setup_logging(yamlpath):
    """Setup logging configuration from yaml file"""
    
    # adapted from https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/

    if os.path.exists(yamlpath):
      with open(yamlpath, 'rt') as f:
        config = yaml.safe_load(f.read())
      logging.config.dictConfig(config)

      logger = logging.getLogger()
      logger.handlers[0].doRollover()
      logger.info('\n--------- Log started on {:s} ---------\n'.format(time.asctime()))
    else:
      warnings.warn('Logging file could not be loaded. Using basic config')
      logging.basicConfig()

def setup_plate_dirs(prefix, plates=[''], silent=False):
  """Creates folders for a plate. Root directory should not exist before running"""

  subdirs = ['config', 'features', 'output', 'preview/pyr', 'preview/well_stitched', 'raw_data', 'segmentation']

  for plate in plates:
    plate_dir = os.path.join(prefix, plate)

    if os.path.exists(plate_dir):
      raise ValueError('Plate dir already exists: '+plate_dir)

    for subdir in subdirs:
      fpath = os.path.join(plate_dir, subdir)
      os.makedirs(fpath)

      if not silent:
        print('Created '+fpath)

def well_to_coord(well, platetype):
  """Convert well to coordinate position. (e.g. A1 -> (0,0) )"""
  try:
    row_idx = get_rows(platetype).index(well.row)
    col_idx = get_cols(platetype).index(well.col)
    return ((row_idx, col_idx))
  except:
    raise ValueError('{:s} not within bounds of {:s} plate'.format(well.printinfo(), platetype))
