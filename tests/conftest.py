"""Provides a generic testing environment for use in tests with custom reset, step and render functions."""
from typing import Any, Dict, Optional, Tuple, Union

import gym
import pytest
from gym import spaces
from gym.core import ActType, ObsType
from gym.envs.registration import EnvSpec


@pytest.fixture
def default_test_env():
    return GenericTestEnv()


class GenericTestEnv(gym.Env):
    """A generic testing environment for use in testing with modified environments are required."""

    def __init__(
        self,
        action_space: Optional[spaces.Space] = None,
        observation_space: Optional[spaces.Space] = None,
        metadata: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
        spec: Optional[EnvSpec] = None,
    ):
        self.metadata = {} if metadata is None else metadata
        self.render_mode = render_mode
        self.spec = (
            spec
            if spec is not None
            else EnvSpec("TestingEnv-v0", "testing-env-no-entry-point")
        )

        self.observation_space = (
            observation_space
            if observation_space is not None
            else spaces.Box(0, 1, (1,))
        )
        self.action_space = (
            action_space if action_space is not None else spaces.Box(0, 1, (1,))
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        # If you need a default working reset function, use `basic_reset_fn` above
        super(GenericTestEnv, self).reset(seed=seed)
        self.observation_space.seed(seed)
        return self.observation_space.sample(), {"options": options}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        return self.observation_space.sample(), 0, False, False, {}

    def render(self):
        # Do nothing
        pass
