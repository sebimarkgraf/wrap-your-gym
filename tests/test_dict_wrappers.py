import gym
import pytest

from wrap_your_gym import ActionDict, ObsDict, ResetInObs, RewardInObs


def test_obs_dict_wrapper(default_test_env):
    wrapped_env = ObsDict(default_test_env, key="image")

    assert isinstance(wrapped_env.observation_space, gym.spaces.Dict)
    obs, info = wrapped_env.reset()
    assert isinstance(obs, dict)
    assert "image" in obs


def test_action_dict_wrapper(default_test_env):
    wrapped_env = ActionDict(default_test_env)

    assert isinstance(wrapped_env.action_space, gym.spaces.Dict)
    act = wrapped_env.action_space.sample()
    assert isinstance(act, dict)
    assert "action" in act
    wrapped_env.reset()
    wrapped_env.step(act)


def test_reward_in_obs(default_test_env):
    wrapped_env = RewardInObs(ObsDict(default_test_env, key="image"))

    obs, info = wrapped_env.reset()
    assert "reward" in obs


def test_reward_fails_when_not_dict(default_test_env):
    with pytest.raises(AssertionError):
        _ = RewardInObs(default_test_env)


def test_reset_in_obs(default_test_env):
    wrapped_env = ResetInObs(ObsDict(default_test_env, key="image"))

    obs, info = wrapped_env.reset()
    assert "reset" in obs


def test_reset_fails_when_not_dict(default_test_env):
    with pytest.raises(AssertionError):
        _ = ResetInObs(default_test_env)
