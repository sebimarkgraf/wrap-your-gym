import torch

from wrap_it.torch_wrappers import ObsPyTorchWrapper


def test_obs_torch_wrapper(default_test_env):
    env = ObsPyTorchWrapper(default_test_env)
    obs, info = env.reset()
    assert isinstance(obs, torch.Tensor)
    act = env.action_space.sample()
    obs, *_ = env.step(act)
    assert isinstance(obs, torch.Tensor)
