import gym
import numpy as np
from gym.spaces import Dict as DictSpace, Box


def _resolve_fetch_env_id(name: str):
    """Map short names to gym IDs (gym==0.19 robotics)."""
    candidates = []
    if name in ["FetchReach", "FetchReach-v1", "FetchReach-v0"]:
        candidates = ["FetchReach-v1", "FetchReach-v0"]
        task = "reach"
    elif name in ["FetchPush", "FetchPush-v1", "FetchPush-v0"]:
        candidates = ["FetchPush-v1", "FetchPush-v0"]
        task = "push"
    elif name in [
        "FetchPickAndPlace", "FetchPickAndPlace-v1", "FetchPickAndPlace-v0",
        "FetchPickAndPlaceDense", "FetchPickAndPlaceDense-v1", "FetchPickAndPlaceDense-v0",
    ]:
        # Dense/normal share observation structure; we still enforce sparse reward in wrapper.
        candidates = ["FetchPickAndPlace-v1", "FetchPickAndPlace-v0"]
        task = "pick_and_place"
    else:
        raise ValueError(f"Unknown Fetch env name: {name}")
    return candidates, task


class TEAFetchAdapter(gym.Wrapper):
    """
    Adapt gym Fetch* (GoalEnv) to the Multiworld-like API your HERReplayBuffer expects:
      - sample_goals(batch_size, keys=set(...))
      - compute_rewards(actions, obs_dict)
    And adapt observation dict to include:
      - observation: original observation vector
      - achieved_goal: state-like achieved goal (we use state vector)
      - desired_goal: state-like desired goal (template state with goal coords overwritten)
      - state_achieved_goal / state_desired_goal (aliases)
    """

    def __init__(self, env, task: str, farthest: bool = False, farthest_k: int = 128):
        super().__init__(env)
        self.task = task
        self.farthest = farthest
        self.farthest_k = int(farthest_k)

        # Goal coordinate indices in the state vector (PlanGAN-style semantics)
        # - reach: gripper position is at state[0:3]
        # - push/pick: object position is at state[3:6]
        if task == "reach":
            self.goal_idx = np.array([0, 1, 2], dtype=np.int64)
        else:
            self.goal_idx = np.array([3, 4, 5], dtype=np.int64)

        # Prefer env-provided threshold if available
        self.distance_threshold = float(getattr(env, "distance_threshold", 0.05))

        # Keep a template state for constructing state-like goals in sample_goals()
        self._template_state = None

        # Track fixed goal for the current episode (state-like)
        self._goal_state = None

        # Build a new observation space consistent with our adapted dict
        obs_dim = int(env.observation_space.spaces["observation"].shape[0])
        low = -np.inf * np.ones(obs_dim, dtype=np.float32)
        high = np.inf * np.ones(obs_dim, dtype=np.float32)

        self.observation_space = DictSpace({
            "observation": Box(low=low, high=high, dtype=np.float32),
            "achieved_goal": Box(low=low, high=high, dtype=np.float32),
            "desired_goal": Box(low=low, high=high, dtype=np.float32),
            "state_achieved_goal": Box(low=low, high=high, dtype=np.float32),
            "state_desired_goal": Box(low=low, high=high, dtype=np.float32),
        })

        # Convenience: expose max_episode_steps if gym spec exists
        self.max_episode_steps = int(getattr(getattr(env, "spec", None), "max_episode_steps", 50) or 50)

    def _state_goal_distance(self, state_vec: np.ndarray, goal_state_vec: np.ndarray) -> float:
        diff = state_vec[self.goal_idx] - goal_state_vec[self.goal_idx]
        return float(np.linalg.norm(diff))

    def _make_goal_state(self, template_state: np.ndarray, desired_goal_3d: np.ndarray) -> np.ndarray:
        g = np.array(template_state, copy=True)
        g[self.goal_idx] = desired_goal_3d
        return g

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        s = np.asarray(obs["observation"], dtype=np.float32)
        dg = np.asarray(obs["desired_goal"], dtype=np.float32)

        if self._template_state is None:
            self._template_state = np.array(s, copy=True)

        # Optionally override to a farthest goal (useful for "hard" eval like your d_test_env)
        if self.farthest:
            # sample k candidate goals, choose farthest from current achieved goal coordinate
            # We try to call the underlying _sample_goal if available.
            candidates = []
            if hasattr(self.env.unwrapped, "_sample_goal"):
                for _ in range(self.farthest_k):
                    candidates.append(np.asarray(self.env.unwrapped._sample_goal(), dtype=np.float32))
            else:
                # fallback: uniform jitter around current desired goal
                for _ in range(self.farthest_k):
                    candidates.append(dg + 0.05 * np.random.randn(*dg.shape).astype(np.float32))

            best_g = dg
            best_d = -1.0
            for c in candidates:
                gs = self._make_goal_state(s, c)
                dist = self._state_goal_distance(s, gs)
                if dist > best_d:
                    best_d = dist
                    best_g = c

            dg = best_g
            # keep underlying env.goal consistent for visualization / internal bookkeeping
            if hasattr(self.env.unwrapped, "goal"):
                try:
                    self.env.unwrapped.goal = np.array(dg, copy=True)
                except Exception:
                    pass

        self._goal_state = self._make_goal_state(s, dg)

        out = {
            "observation": np.array(s, copy=True),
            "achieved_goal": np.array(s, copy=True),
            "desired_goal": np.array(self._goal_state, copy=True),
            "state_achieved_goal": np.array(s, copy=True),
            "state_desired_goal": np.array(self._goal_state, copy=True),
        }
        return out

    def step(self, action):
        obs, _, done_env, info = self.env.step(action)

        s = np.asarray(obs["observation"], dtype=np.float32)
        dist = self._state_goal_distance(s, self._goal_state)
        is_success = float(dist < self.distance_threshold)

        # Sparse reward: -1 per step, 0 on success (matches your stated setting)
        reward = 0.0 if is_success > 0.5 else -1.0

        # We do NOT rely on env's done except for time-limit; success termination is handled in your loop,
        # but returning a sensible done is still helpful.
        done = bool(done_env) or bool(is_success > 0.5)

        info = dict(info or {})
        info["distance"] = dist
        info["is_success"] = is_success
        # keep backward-compatible key if your logging expects it
        info["xy-distance"] = dist

        out = {
            "observation": np.array(s, copy=True),
            "achieved_goal": np.array(s, copy=True),
            "desired_goal": np.array(self._goal_state, copy=True),
            "state_achieved_goal": np.array(s, copy=True),
            "state_desired_goal": np.array(self._goal_state, copy=True),
        }
        return out, reward, done, info

    # ===== Multiworld-like API for your HERReplayBuffer =====
    def sample_goals(self, batch_size: int, keys=None):
        """
        Return a dict of goal vectors for requested keys.
        We construct state-like goals using a stored template state + sampled 3D goals.
        """
        if keys is None:
            keys = {"desired_goal", "state_desired_goal"}
        if isinstance(keys, (list, tuple)):
            keys = set(keys)

        if self._template_state is None:
            # ensure template exists
            self.reset()

        # sample raw desired goals
        goals_3d = []
        if hasattr(self.env.unwrapped, "_sample_goal"):
            for _ in range(int(batch_size)):
                goals_3d.append(np.asarray(self.env.unwrapped._sample_goal(), dtype=np.float32))
        else:
            # fallback: small jitter around current goal
            cur = self.env.unwrapped.goal if hasattr(self.env.unwrapped, "goal") else np.zeros(3, dtype=np.float32)
            for _ in range(int(batch_size)):
                goals_3d.append(np.asarray(cur, dtype=np.float32) + 0.05 * np.random.randn(3).astype(np.float32))
        goals_3d = np.stack(goals_3d, axis=0)

        goal_states = np.stack([self._make_goal_state(self._template_state, g) for g in goals_3d], axis=0).astype(np.float32)

        out = {}
        for k in keys:
            out[k] = goal_states
        return out

    def compute_rewards(self, actions, obs_dict):
        """
        Vectorized sparse reward based on distance between state-like achieved_goal and desired_goal.
        Signature matches your HERReplayBuffer usage.
        """
        ag = np.asarray(obs_dict["achieved_goal"], dtype=np.float32)
        dg = np.asarray(obs_dict["desired_goal"], dtype=np.float32)
        diff = ag[:, self.goal_idx] - dg[:, self.goal_idx]
        dist = np.linalg.norm(diff, axis=-1)
        r = np.where(dist < self.distance_threshold, 0.0, -1.0).astype(np.float32)
        return r.reshape(-1, 1)


def make_fetch_env(name: str, seed: int = 0, farthest: bool = False):
    # Ensure robotics envs are registered (gym==0.19)
    import gym.envs.robotics  # noqa: F401

    candidates, task = _resolve_fetch_env_id(name)
    last_err = None
    env = None
    for env_id in candidates:
        try:
            env = gym.make(env_id)
            break
        except Exception as e:
            last_err = e
            env = None
    if env is None:
        raise RuntimeError(f"Failed to make Fetch env from {candidates}. Last error: {last_err}")

    try:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass

    wrapped = TEAFetchAdapter(env, task=task, farthest=farthest)
    return wrapped
