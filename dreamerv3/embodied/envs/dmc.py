import functools
import glob
import os

import embodied
import numpy as np


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      locom_rodent=1,
      quadruped=2,
  )

  def __init__(
      self, 
      env, 
      seed=0,
      repeat=1, 
      render=True, 
      size=(64, 64), 
      camera=-1, 
      back_type=None, 
      back_path=None,
      total_frames=1000,
      grayscale=False,
      ):
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'

    if back_type == 'video':
      files = glob.glob(os.path.expanduser(back_path))
      assert len(files), "Pattern {} does not match any files".format(back_path)
      self._bg = embodied.background.RandomVideoSource(
        size, files, seed=seed, grayscale=grayscale, total_frames=total_frames
      )
    elif back_type == 'pickle':
      files = glob.glob(os.path.expanduser(back_path))
      assert len(files), "Pattern {} does not match any files".format(back_path)
      self._bg = embodied.background.RandomPickleSource(files, seed=seed)
    elif back_type == 'noise':
      self._bg = embodied.background.RandomNoise(
        size, seed=seed, grayscale=grayscale, total_frames=total_frames
      )
    else:
      self._bg = None

    if self._bg:
      from dm_control.utils import io as resources
      import dm_control.suite
      _SUITE_DIR = os.getcwd()
      _FILENAMES = [
        "dreamerv3/embodied/envs/dmc_xml/materials.xml",
        "dreamerv3/embodied/envs/dmc_xml/skybox.xml",
        "dreamerv3/embodied/envs/dmc_xml/visual.xml",
      ]
      ASSETS = {filename: resources.GetResource(os.path.join(_SUITE_DIR, filename)) for filename in _FILENAMES}
      def read_model(model_filename):
        """Reads a model XML file and returns its contents as a string."""
        return resources.GetResource(model_filename)
      def get_model_and_assets():
        """Returns a tuple containing the model XML string and a dict of assets."""
        return read_model(os.path.join(_SUITE_DIR, "dreamerv3/embodied/envs/dmc_xml/walker.xml")), ASSETS
      dm_control.suite.walker.get_model_and_assets = get_model_and_assets
    
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        from dm_control import manipulation
        env = manipulation.load(task + '_vision')
      elif domain == 'locom':
        from dm_control.locomotion.examples import basic_rodent_2020
        env = getattr(basic_rodent_2020, task)()
      else:
        from dm_control import suite
        env = suite.load(domain, task)

    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._render = render
    self._size = size
    self._camera = camera

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    if action['reset'] and self._bg:
      self._bg.build_arr()
      self._bg.current_idx = 0
      self._bg.reset()
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    if self._render:
      obs['image'] = self.render()
    return obs

  def render(self):
    img = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    if self._bg is not None:
      mask = np.all(img == 0, axis=-1)
      bg = self._bg.get_image()
      img[mask] = bg[mask]
    return img
