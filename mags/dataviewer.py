"""
Visualize data
"""

import collections
import cv2
from datareader import DataReader, ImageInfo, Well
import datetime
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils import helperfuns, imfuns

class DataViewer:

  def __init__(self, dreader):
    self.colormaps = {
      'blue': sns.cubehelix_palette(256, start=2.92, rot=0, dark=0, light=.65, gamma=1, hue=3, reverse=True, as_cmap=True),
      'green': sns.cubehelix_palette(256, start=2, rot=0, dark=0, light=.65, gamma=1, hue=3, reverse=True, as_cmap=True),
      'red': sns.cubehelix_palette(256, start=0.89, rot=0, dark=0, light=.55, gamma=1, hue=3, reverse=True, as_cmap=True),
      'pink': sns.cubehelix_palette(256, start=0.55, rot=0, dark=0, light=.85, gamma=1, hue=2, reverse=True, as_cmap=True),
      None: None
      }

    self.dreader = dreader
    
    row_lst = helperfuns.get_rows(self.dreader.plateinfo.platetype)
    self.row_to_num = {r: i for i, r in enumerate(row_lst)}
    self.num_to_row = {i: r for i, r in enumerate(row_lst)}
    
    self.viz_settings = self.dreader.viz_settings

    # default plotting settings
    self.dpi = 1000
    self.showfig = True
    self.savefig = False
    self.facecolor = 'black'
    self.treatment_label = False
    self.hspace = 0.2
    self.welltitle_size = 5
    self.figtitle_size = 5

    plt.rcParams.update({'axes.titlesize': self.welltitle_size})
    plt.rcParams.update({'figure.titlesize': self.figtitle_size})

  def generic_viewer(self, im_dic, plot_coords=[]):

    if plot_coords:
      num_rows = max([x[0] for x in plot_coords])+1
      num_cols = max([x[1] for x in plot_coords])+1
    else:
      num_ims = len(im_dic.keys())
      num_rows = max(1, math.floor(math.sqrt(num_ims)))
      num_cols = math.ceil(num_ims / num_rows)

      plot_rows = np.repeat(range(num_rows), num_cols)[:num_ims]
      plot_cols = (list(range(num_cols))*num_rows)[:num_ims]
      plot_coords = list(zip(plot_rows, plot_cols))
    
    im_coords = {c: i for i, c in enumerate(plot_coords)}

    im_sz = list(im_dic.values())[0]['composite'].shape

    fig, axes = plt.subplots(num_rows, num_cols, facecolor=self.facecolor, sharex='all', sharey='all')
    if not isinstance(axes, np.ndarray):
      axes = np.array([[axes]])

    for coord, ax in np.ndenumerate(axes):

      if num_rows == 1:
        coord = (0,)+coord
      elif num_cols == 1:
        coord = coord+(0,)

      if coord in im_coords:
        im_num = im_coords[coord]
        key, ims = list(im_dic.items())[im_num]

        if self.treatment_label:
          title = ims['name']
        else:
          title = key

        ax.imshow(ims['composite'], zorder=1)
        ax.set_title(title, color='w')
        ax.axis('off')
      else:
        ax.imshow(np.zeros(im_sz), zorder=1) #np.zeros(im_sz)[:,:,0:3]
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=self.hspace)

    fig.suptitle(self.dreader.exp_name, color='w')

    if self.savefig:
      outpath = os.path.join(self.dreader.output_path, '{time_now:%Y%m%d-%H%M%S}.png'.format(time_now=datetime.datetime.now()))
      plt.savefig(outpath, dpi=self.dpi, facecolor=self.facecolor)
      print('Image saved at: '+outpath)
    if self.showfig:
      plt.show()

  def read_ims(self, iminfo, z=None, im_type='well'):
    """im_type: 'well' or 'raw'"""

    ims = {}

    if not isinstance(iminfo, ImageInfo):
      iminfo = ImageInfo(col=iminfo.col, row=iminfo.row, fld=0)

    for marker, viz_info in self.viz_settings.items():

      if z is None:
        z = viz_info['zplanes']

      im = self.dreader.get_max_im(iminfo, None, z_lst=z, marker=marker, data_source=im_type)
      im = imfuns.convert_to_uint8(im)

      # scale image
      im_scaled = imfuns.im_linscale(im, viz_info['thresh_upper'], viz_info['thresh_lower'])

      # colorize image
      cmap = self.colormaps[viz_info['color']]
      if cmap is None:
        ims[marker] = im_scaled
      else:
        ims[marker] = cmap(im_scaled, bytes=True)   

    return ims

  def plate_viewer(self, wells=None, plate_grid=True):
    if wells is None:
      wells = self.dreader.plateinfo.wells

    im_dic = collections.OrderedDict()
    for well in wells:
      key = well.row+str(well.col)

      im_dic[key] = {} 

      im_dic[key]['name'] = ' + '.join(self.dreader.plateinfo.platemap[helperfuns.well_to_coord(well, self.dreader.plateinfo.platetype)])

      im_dic[key]['ims'] = self.read_ims(well)

      im_dic[key]['composite'] = imfuns.blend_ims(list(im_dic[key]['ims'].values()))

    if plate_grid:
      row_min = min([self.row_to_num[w.row] for w in wells])
      col_min = min([w.col for w in wells])

      coords = [(self.row_to_num[w.row]-row_min, int(w.col)-col_min) for w in wells]

      self.generic_viewer(im_dic, plot_coords=coords)
    else:
      self.generic_viewer(im_dic)

  def z_stack_viewer(self, iminfo, im_type='well'):

    im_dic = collections.OrderedDict()

    for z in self.dreader.plateinfo.z_lst:
      key = 'Z = ' + str(z)

      # colorize image
      im_dic[key] = self.read_ims(iminfo, z=[z], im_type=im_type)

      # create composite images 
      im_dic[key]['composite'] = imfuns.blend_ims(list(im_dic[key].values()))

    self.generic_viewer(im_dic)
  

  

