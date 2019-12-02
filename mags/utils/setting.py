import sys
import yaml

with open('config/settings.yaml') as f:
  settings = yaml.safe_load(f.read())

def get_path_for_os(path_dic):
  if sys.platform == 'darwin':
    return path_dic['macosx']
  elif sys.platform == 'linux':
    return path_dic['linux']
  else:
    raise ValueError('Unknown operating system '+sys.platform)

max_pixel_val_dic = {
    'uint12': 4095
  }


PLATE_TEMPLATE = {
  '96_well': {
    'num_rows': 8,
    'num_cols': 12,
    'rows': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'cols': list(range(1, 13))
    },
  '384_well': {
    'num_rows': 16,
    'num_cols': 24
    }
}

FIJI = {
  'path': get_path_for_os(settings['fiji_path']),
  'batch_tiff_save_macro': settings['batch_tiff_macro']
}

class BFConvertSettings:
  def __init__(self, timelapse=False):
    self.path = get_path_for_os(settings['bfconvert_path'])

    if timelapse: 
      self.format = settings['bfconvert_formatter_time']
      self.fname_pattern = settings['bfconvert_tiff_pattern_timelapse']
    else:
      self.format = settings['bfconvert_formatter']
      self.fname_pattern = settings['bfconvert_tiff_pattern']

DROPFILE = settings['dropfile']

SEGDROPFILE = settings['segdropfile']

IMPATH = settings['impath']

with open('config/default_seg_params.yaml') as f:
  DEFAULT_SEG_PARAMS = yaml.safe_load(f.read())
