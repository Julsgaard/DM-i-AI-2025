# racecar_env.py
# A tiny Gymnasium-compatible wrapper around your simulator so we can train PPO fast.

import numpy as np
from gymnasium import Env, spaces
import pygame

FRAME_SKIP = 10              # match your API batch size; lower = more reactive
CRASH_FRONT_THRESH = 70.0   # if min(front, diagonals) < this -> terminate early
DANGER_NEAR = 200.0         # near zone
DANGER_MED  = 350.0         # medium zone

# Import the simulator module *as a module* so STATE is always the live object
try:
    import src.game.core as core  # typical project layout
except ImportError:
    import core as core           # fallback if core.py is at project root

# Keep sensor order stable so the observation vector is consistent
SENSOR_ORDER = [
    "left_side", "left_side_front", "left_front", "front_left_front",
    "front", "front_right_front", "right_front", "right_side_front",
    "right_side", "right_side_back", "right_back", "back_right_back",
    "back", "back_left_back", "left_back", "left_side_back",
]

ACTIONS = ["ACCELERATE", "DECELERATE", "STEER_LEFT", "STEER_RIGHT", "NOTHING"]


def _read_sensors():
    vals = {}
    for i, s in enumerate(core.STATE.sensors):
        name = getattr(s, "name", f"sensor_{i}")
        v = 1000.0
        if hasattr(s, "reading"):
            try:
                v = float(getattr(s, "reading"))
            except Exception:
                v = 1000.0
        elif hasattr(s, "sensor_strength"):
            v = float(getattr(s, "sensor_strength"))
        vals[name] = v
    return vals


def _tick_sim(action_str: str) -> float:
    """
    Advance one simulation step reusing your core loop pieces.
    Returns: delta distance progressed this step.
    """
    core.handle_action(action_str)

    old_dist = core.STATE.distance
    core.STATE.distance += core.STATE.ego.velocity.x

    core.update_cars()
    core.remove_passed_cars()
    core.place_car()

    # Update sensors after world update
    for sensor in core.STATE.sensors:
        sensor.update()

    # Collisions
    for car in core.STATE.cars:
        if car is not core.STATE.ego and core.intersects(core.STATE.ego.rect, car.rect):
            core.STATE.crashed = True
    for wall in core.STATE.road.walls:
        if core.intersects(core.STATE.ego.rect, wall.rect):
            core.STATE.crashed = True

    return core.STATE.distance - old_dist


class RaceCarEnv(Env):
    """
    Observation (float32, shape=18):
      - 16 normalized sensor distances in fixed order (0..1, where 1 ~= clear)
      - normalized vx in [0..1] (vx/20)
      - normalized vy in [0..1] (map [-5..5] -> [0..1])

    Action (Discrete 5): index into ACTIONS list.
    """
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 1800):
        super().__init__()
        self.max_steps = max_steps
        self.steps = 0

        # 16 sensors + 2 velocity terms
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

    def _obs(self) -> np.ndarray:
        svals = _read_sensors()
        sensors = [svals.get(k, 1000.0) for k in SENSOR_ORDER]
        sensors = np.clip(np.array(sensors, dtype=np.float32) / 1000.0, 0.0, 1.0)

        vx = float(core.STATE.ego.velocity.x)
        vy = float(core.STATE.ego.velocity.y)

        vx_n = np.clip(vx / 20.0, 0.0, 1.0)
        vy_n = np.clip((vy + 5.0) / 10.0, 0.0, 1.0)  # [-5..5] -> [0..1]

        return np.concatenate([sensors, [vx_n, vy_n]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if not pygame.get_init():
            pygame.init()
        core.initialize_game_state(api_url="http://localhost:9052/predict", seed_value=seed, sensor_removal=0)
        core.STATE.crashed = False
        core.STATE.distance = 0.0
        core.STATE.elapsed_game_time = 0
        core.STATE.ticks = 0
        self.steps = 0
        # clear shaping history
        for a in ("_last_min_front", "_last_left", "_last_right"):
            if hasattr(self, a): delattr(self, a)
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        act_idx = int(action)
        act_idx = max(0, min(act_idx, len(ACTIONS) - 1))
        act_str = ACTIONS[act_idx]

        total_reward = 0.0
        terminated = False

        # how many sim ticks per chosen action; keep this = BATCH_LEN at serve time
        skip = FRAME_SKIP if "FRAME_SKIP" in globals() else 1

        for _ in range(skip):
            delta = _tick_sim(act_str)

            # --- read sensors ---
            s = _read_sensors()
            # front sector
            front = float(s.get("front", 1000.0))
            flf = float(s.get("front_left_front", 1000.0))
            frf = float(s.get("front_right_front", 1000.0))
            l_front = float(s.get("left_front", 1000.0))
            r_front = float(s.get("right_front", 1000.0))
            # sides (incl. side-back to estimate clearance corridor)
            left = float(s.get("left_side", 1000.0))
            right = float(s.get("right_side", 1000.0))
            lsb = float(s.get("left_side_back", 1000.0))
            rsb = float(s.get("right_side_back", 1000.0))
            # rear cluster
            back = float(s.get("back", 1000.0))
            blb = float(s.get("back_left_back", 1000.0))
            brb = float(s.get("back_right_back", 1000.0))
            l_back = float(s.get("left_back", 1000.0))
            r_back = float(s.get("right_back", 1000.0))

            min_front = min(front, flf, frf, l_front, r_front)
            min_back = min(back, blb, brb, l_back, r_back)
            min_left = min(left, l_front, lsb)
            min_right = min(right, r_front, rsb)

            vx = float(core.STATE.ego.velocity.x)
            vy = float(core.STATE.ego.velocity.y)

            # --------- INIT r FIRST (fixes UnboundLocalError) ----------
            # base reward: very small forward progress, the rest is safety
            r = 0.05 * float(delta)

            # --- side collision awareness (uses real 120px limit + buffer) ---
            SIDE_COLLISION = 120.0
            SIDE_SAFE = 180.0

            def side_danger(d):
                if d <= SIDE_COLLISION: return 1.0
                if d >= SIDE_SAFE:      return 0.0
                return 1.0 - (d - SIDE_COLLISION) / (SIDE_SAFE - SIDE_COLLISION)

            sd_left = side_danger(min_left)
            sd_right = side_danger(min_right)

            # penalize tight sides, stronger when very tight
            r -= 1.2 * max(sd_left, sd_right)

            # never reward steering into a tight side; penalize it
            if act_str == "STEER_LEFT" and sd_left > 0.4: r -= 0.8
            if act_str == "STEER_RIGHT" and sd_right > 0.4: r -= 0.8

            # small shaping bonus for increasing side clearance (encourage preparing dodges early)
            if not hasattr(self, "_last_left"):  self._last_left = min_left
            if not hasattr(self, "_last_right"): self._last_right = min_right
            r += 0.002 * max(min_left - self._last_left, 0.0)
            r += 0.002 * max(min_right - self._last_right, 0.0)
            self._last_left, self._last_right = min_left, min_right

            # --- time-to-collision style danger (front) ---
            eps = 1e-3
            ttc = min_front / max(vx, eps)  # small ttc = very dangerous
            # danger in [0..1], where ttc <= 30 ticks => ~1, ttc >= 90 => ~0
            danger = 1.0 - np.clip((ttc - 30.0) / (90.0 - 30.0), 0.0, 1.0)
            r -= 2.0 * danger  # strong: prioritize not crashing

            # penalize getting closer to obstacles ahead (delta clearance)
            if not hasattr(self, "_last_min_front"):
                self._last_min_front = min_front
            r -= 0.002 * max(self._last_min_front - min_front, 0.0)
            self._last_min_front = min_front

            # lane centering by usable side clearance
            center_err = abs(min_left - min_right) / 1000.0
            r -= 0.05 * center_err

            # smooth lateral motion
            r -= 0.02 * abs(vy)

            # speed management in danger zones
            if min_front < 350.0:
                r -= 0.06 * max(vx - 6.0, 0.0)  # too fast into front danger
            if min_back < 350.0:
                r -= 0.05 * max(4.0 - vx, 0.0)  # don't crawl if tailgated

            # action-aware shaping (only meaningful when danger)
            if min_front < 350.0:
                if act_str == "ACCELERATE": r -= 0.8
                if act_str == "DECELERATE": r += 0.6
                if min_left < min_right and act_str == "STEER_RIGHT": r += 0.4
                if min_right < min_left and act_str == "STEER_LEFT":  r += 0.4

            if min_back < 350.0:
                if act_str == "ACCELERATE": r += 0.2
                if min_left > min_right and act_str == "STEER_LEFT":  r += 0.15
                if min_right > min_left and act_str == "STEER_RIGHT": r += 0.15

            # tiny speed bonus only when fully safe front & back
            if min_front >= 400.0 and min_back >= 350.0:
                r += 0.002 * max(vx - 8.0, 0.0)

            # hard termination
            if min_front < 60.0:
                r -= 100.0
                terminated = True
            if core.STATE.crashed:
                r -= 100.0
                terminated = True

            total_reward += r
            if terminated:
                break

        truncated = self.steps >= self.max_steps
        return self._obs(), float(total_reward), terminated, truncated, {}


