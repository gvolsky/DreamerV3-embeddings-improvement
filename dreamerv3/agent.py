import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
df = lambda d: tree_map(lambda x: x[:, 1:], d)
dl = lambda d: tree_map(lambda x: x[:, :-1], d)
USING_HIST = ['bisim2']
USING_ACT = ['bisim1', 'bisim2', 'bisim3', 'bisim4']

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = expl_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    print(f"act_space: {self.act_space.shape}")
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')
    }
    self.enc_loss = self.config.enc_loss.impl
    print(f"ENC LOSS IMPL: {self.enc_loss}")
    if self.enc_loss in USING_ACT:
      self.act = nets.MLP(
        None, layers=2, units=4096, activations=['silu', 'none'], inputs=['tensor'], name='act'
      )
      self.repr = nets.MLP(
        None, layers=2, units=4096, activations=['silu', 'none'], inputs=['tensor'], name='repr'
      )

    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    if self.enc_loss in USING_ACT:
      modules.extend([self.act, self.repr])
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    key = nj.rng()

    if self.enc_loss in USING_ACT:
      embed_data = jnp.concatenate([sg(prior['deter']), embed], -1) if self.enc_loss in USING_HIST \
                                                                    else embed
      act_net = self.act(sg(dl(data['action'])))
      repr_net = self.repr(dl(embed_data))
      repr_out = act_net * repr_net
      losses['rpred'] = jnp.mean(jnp.abs(repr_out - sg(df(embed))))

    if self.enc_loss == 'bisim1':
      new_repr = sg(act_net) * repr_net
      idxs = jax.random.permutation(nj.rng(), self.config.batch_size)
      nrepr_dist = jnp.mean(jnp.abs(new_repr - sg(new_repr[idxs])), axis=-1)
      reward = df(data['reward'])
      r_dist = jnp.abs(reward - sg(reward[idxs])) 
      act = sg(self.act(jax.random.uniform(nj.rng(), shape=dl(data['action']).shape, minval=-1.0, maxval=1.0)))
      nnrepr_transform = sg(self.repr(sg(new_repr)))
      nnrepr1 = nnrepr_transform * act
      nnrepr2 = nnrepr1[idxs]
      nnrepr_dist = jnp.mean(jnp.abs(sg(nnrepr1) - sg(nnrepr2)), axis=-1)
      bisim = r_dist + self.config.enc_loss.disc * nnrepr_dist
      losses['enc'] = jnp.mean(jnp.square(nrepr_dist - bisim))

    elif self.enc_loss == 'bisim2':
      new_repr = sg(act_net) * repr_net
      idxs = jax.random.permutation(nj.rng(), self.config.batch_size)
      nrepr_dist = jnp.mean(jnp.abs(new_repr - new_repr[idxs]), axis=-1)
      reward = dl(data['reward'])
      r_dist = jnp.abs(reward - sg(reward[idxs])) 
      act = self.act(jax.random.uniform(key, shape=dl(data['action']).shape, minval=-1.0, maxval=1.0))
      nnrepr_transform = self.repr(embed_data[:, 1:])
      nnrepr1, nnrepr2 = sg(nnrepr_transform * act), sg(nnrepr_transform[idxs] * act)
      nnrepr_dist = jnp.mean(jnp.abs(nnrepr1 - nnrepr2), axis=-1)
      bisim = r_dist + self.config.enc_loss.disc * nnrepr_dist
      losses['enc'] = jnp.mean(jnp.square(nrepr_dist - bisim))

    elif self.enc_loss == 'bisim3':
      new_repr = sg(act_net) * repr_net
      idxs = jax.random.permutation(nj.rng(), self.config.batch_size)
      # reward = data['reward']
      # r_dist = jnp.abs(reward - sg(reward[idxs])) 
      ract = jax.random.uniform(key, shape=dl(prev_actions).shape, minval=-1.0, maxval=1.0)
      rand_repr = self.act(ract) * repr_net
      rand_repr_dist = jnp.mean(jnp.abs(rand_repr - sg(rand_repr[idxs])), axis=-1)
      new_embed = jnp.concatenate([embed[:, 0, None], rand_repr], 1)
      new_act = jnp.concatenate([prev_action[:, None], ract], 1)
      npost, nprior = self.rssm.observe(sg(new_embed), sg(new_act), data['is_first'], prev_latent)
      nfeats = {**npost, 'embed': new_embed}
      nreward = self.heads['reward'](sg(nfeats)).mean()
      r_dist = jnp.abs(sg(nreward) - sg(nreward[idxs]))
      mean = self.rssm.get_dist(sg(nprior), idxs, get_mean=True)
      t_dist = 0.5 * self.rssm.get_dist(sg(nprior)).kl_divergence(mean) + \
               0.5 * self.rssm.get_dist(sg(nprior), idxs).kl_divergence(mean)
      bisim = r_dist + self.config.enc_loss.disc * t_dist
      losses['enc'] = jnp.mean(jnp.square(rand_repr_dist - df(bisim)))

    elif self.enc_loss == 'bisim4':
      act_transform = self.act(prev_actions[:, :-1])
      repr_transform = self.repr(embed)
      new_repr = act_transform * dl(repr_transform)
      losses['rpred'] = jnp.mean(jnp.abs(new_repr - sg(df(embed))))
      idxs = jax.random.permutation(key, self.config.batch_size)
      reward = data['reward']
      r_dist = jnp.abs(reward - sg(reward[idxs])) 
      ract = jax.random.uniform(key, shape=prev_actions.shape, minval=-1.0, maxval=1.0)
      rand_repr = self.act(ract) * repr_transform
      rand_repr_dist = jnp.mean(jnp.abs(rand_repr - sg(rand_repr[idxs])), axis=-1)
      _, nprior = self.rssm.observe(embed, ract, jnp.zeros_like(data['is_first']), prev_latent)
      mean = self.rssm.get_dist(sg(nprior), idxs, get_mean=True)
      t_dist = 0.5 * self.rssm.get_dist(sg(nprior)).kl_divergence(mean) + \
               0.5 * self.rssm.get_dist(sg(nprior), idxs).kl_divergence(mean)
      bisim = r_dist + self.config.enc_loss.disc * t_dist
      losses['enc'] = jnp.mean(jnp.square(rand_repr_dist - bisim))

    elif self.enc_loss == 'bisim':
      idxs = jax.random.permutation(key, self.config.batch_size)
      z_dist = jnp.mean(jnp.abs(embed - embed[idxs]), axis=-1)
      r_dist = jnp.abs(data['reward'] - sg(data['reward'][idxs])) 
      if self.rssm._classes:
        mean = self.rssm.get_dist(sg(prior), idxs, get_mean=True)
        t_dist = 0.5 * self.rssm.get_dist(sg(prior)).kl_divergence(mean) + \
                 0.5 * self.rssm.get_dist(sg(prior), idxs).kl_divergence(mean)
      else:
        mean, std = prior['mean'], prior['std']
        t_dist = jnp.mean(jnp.sqrt(
          jnp.square(sg(mean[idxs]) - sg(mean)) + jnp.square(sg(std[idxs]) - sg(std))
        ), axis=-1) 
      bisim = r_dist + self.config.enc_loss.disc * t_dist
      losses['enc'] = jnp.mean(jnp.square(z_dist - bisim))

    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]
