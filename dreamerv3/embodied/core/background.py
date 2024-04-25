import numpy as np
import cv2
import pickle
import random

from itertools import cycle

def read_video(video_path, grayscale=False):
  cap = cv2.VideoCapture(video_path)
  
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
  if grayscale:
    all_frames = np.zeros((num_frames, height, width), dtype=np.uint8)
  else:
    all_frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
  
  i = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      all_frames[i] = frame
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

  def reset(self):
    """ Called when an episode ends. """
    pass

# FIXME: fix random state
class RandomVideoSource(ImageSource):
  def __init__(self, shape, filelist, seed=0, total_frames=None, grayscale=False):
    """
    Args:
      shape: [h, w]
      filelist: a list of video files
    """
    self.seed = seed
    print(f"BACKGR SEED: {self.seed}")
    np.random.seed(self.seed)
    random.seed(self.seed)
    self.grayscale = grayscale
    self.total_frames = total_frames
    self.shape = shape
    self.filelist = filelist
    self.build_arr()
    self.current_idx = 0
    self.reset()

  def build_arr(self):
    if not self.total_frames:
      self.total_frames = 0
      self.arr = None
      random.shuffle(self.filelist)
      for fname in self.filelist:
        frames = read_video(fname, self.grayscale)
        local_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
        for i in range(frames.shape[0]):
          local_arr[i] = cv2.resize(frames[i], (self.shape[1], self.shape[0]))
        if self.arr is None:
          self.arr = local_arr
        else:
          self.arr = np.concatenate([self.arr, local_arr], 0)
        self.total_frames += local_arr.shape[0]
    else:
      self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
      total_frame_i = 0
      file_i = 0
      while total_frame_i < self.total_frames:
        if file_i % len(self.filelist) == 0: 
          random.shuffle(self.filelist)
        file_i += 1
        fname = self.filelist[file_i % len(self.filelist)]
        frames = read_video(fname, self.grayscale)
        for frame_i in range(frames.shape[0]):
          if total_frame_i >= self.total_frames: 
            break
          if self.grayscale:
            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]))[..., None]
          else:
            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0])) 
          total_frame_i += 1

  def reset(self):
    self._loc = np.random.randint(0, self.total_frames)

  def get_image(self):
    img = self.arr[self._loc % self.total_frames]
    self._loc += 1
    return img
  

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
    self.arr = cycle(self.rand_gen.integers(
      0, 256, (self.total_frames, *self.shape, self.channels), dtype=np.uint8
    ))

  def get_image(self):
    return next(self.arr)