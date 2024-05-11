import numpy as np
import cv2
import pickle
import random

from itertools import cycle

def read_video(video_path, shape, grayscale=False):
  cap = cv2.VideoCapture(video_path)
  
  # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  if grayscale:
    all_frames = np.zeros((num_frames, *shape), dtype=np.uint8)
  else:
    all_frames = np.zeros((num_frames, *shape, 3), dtype=np.uint8)
  
  i = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      all_frames[i] = cv2.resize(frame, shape)
      i += 1
    else:
      break

  cap.release()
  return all_frames


class ImageSource(object):
  """
  Source of natural images to be added to a simulated environment.
  """
  def get_image(self):
    """
    Returns:
        an RGB image of [h, w, 3] with a fixed shape.
    """
    pass

  def build_arr(self):
    """ Called when an episode ends. """
    pass


# FIXME: fix random state
class RandomVideoSource(ImageSource):
  def __init__(self, shape, filelist, seed=0, grayscale=False):
    """
    Args:
      shape: [h, w]
      filelist: a list of video files
    """
    self.seed = seed
    print(f"BACKGR SEED: {self.seed}")
    self.rand_gen = random.Random(seed)
    self.grayscale = grayscale
    self.shape = shape
    self.filelist = filelist
    self.build_arr()

  def build_frames(self):
    frames = []
    while len(frames) == 0:
      fname = self.rand_gen.choice(self.filelist)
      frames = read_video(fname, self.shape, self.grayscale)
    return cycle(frames)
  
  def build_arr(self):
    self.backgr = self.build_frames()
    self.grid = self.build_frames()

  def get_images(self):
    return next(self.backgr), next(self.grid)
  

class RandomPickleSource(ImageSource):
  def __init__(self, filelist, seed=0):
    """
    Args:
      shape: [h, w]
      filelist: a list of pickle files
    """
    self.filelist = filelist
    print(f"BACKGR SEED: {seed}")
    self.rand_gen = random.Random(seed)
    self.build_arr()

  def build_arr(self):
    fname = self.rand_gen.choice(self.filelist)
    with open(fname, 'rb') as f:
      self.arr = cycle(pickle.load(f))

  def get_image(self):
    return next(self.arr)
  
  
class RandomNoise(ImageSource):
  def __init__(self, shape, seed=0, total_frames=None, grayscale=False):
    """
    Args:
      shape: [h, w]
    """
    self.channels = 1 if grayscale else 3
    self.total_frames = total_frames if total_frames else 500
    self.shape = shape
    print(f"BACKGR SEED: {seed}")
    self.rand_gen = np.random.default_rng(seed)
    self.build_arr()

  def build_arr(self):
    self.backgr = cycle(self.rand_gen.integers(
      0, 256, (self.total_frames, *self.shape, self.channels), dtype=np.uint8
    ))
    self.grid = cycle(self.rand_gen.integers(
      0, 256, (self.total_frames, *self.shape, self.channels), dtype=np.uint8
    ))

  def get_images(self):
    return next(self.backgr), next(self.grid)