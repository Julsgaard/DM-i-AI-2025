import random
from typing import List
import math


'''
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
'''

# def return_action(state):
#     # Returns a list of actions
#     actions = []
#     action_choices = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
#     for _ in range(1):
#         actions.append(random.choice(action_choices))
#     # print(actions)
#     return actions




# TODO: LANE SWITCHER

# Triggers on front sensor < BLOCK_THR
# Chooses the clearer side (right vs left) using two forward-diagonal sensors.
# Switch = N steps to side (A) + N steps back (B) to re-straighten, then IDLE.

# Tiny state (persists across calls)
MODE = "IDLE" # IDLE | RIGHT_A | RIGHT_B | LEFT_A | LEFT_B
STEPS_LEFT = 0

# Remember last distances + trend counters
PREV_FRONT = None
PREV_BACK = None
TREND_FRONT = 0
TREND_BACK = 0
# Trend sensitivity
TREND_EPS = 5.0 # pixels change to consider "meaningful"
TREND_MIN = 2 # need this many consecutive "getting closer" ticks to trigger

# Tunables
BATCH_SIZE = 8
N_SWITCH = 48
BLOCK_THR = 950.0 # obstacle if sensor < this
TARGET_VX = 20
VX_BAND = 0.15
BASE_TARGET_VX = 10     # start slow
MAX_TARGET_VX  = 999    # The maximum target VX
RAMP_PER_TICK  = 0.05   # vx gained per tick
CURRENT_TARGET_VX = BASE_TARGET_VX  # computed each call

REL_TOL = 1e-6
ABS_TOL = 1e-3

# TODO: It needs to gradually accelerate infinitely. This is because the cars are spawning at the same speed as the race-car.
#  If i only accelerate it will be too fast to dodge, but if i gradually accelerate it might be enough to dodge.
#  I think that even though you are moving at 100 velocity the cars will spawn at the same speed.
#  This is confirmed by what it says in place_car function in core.py

# Sensor groups (use the names you already have in your sim payload)
LEFT_FWD   = ("front_left_front", "left_side_front")
RIGHT_FWD  = ("front_right_front", "right_side_front")
LEFT_BACK  = ("back_left_back", "left_side_back")
RIGHT_BACK = ("back_right_back", "right_side_back")


def _s(state, name, default=1000.0):
    return float((state.get("sensors") or {}).get(name, default) or default)

def _min_of(state, names):
    sens = state.get("sensors") or {}
    vals = [float(sens.get(n, 1000.0) or 1000.0) for n in names if n in sens]
    return min(vals) if vals else 1000.0

def _maybe_start_switch(state):
    """
    Trigger a lane switch if front or back is blocked.
    Choose clearer side: compare left vs right groups (front case uses FWD groups,
    back case uses BACK groups). No randomness.
    """
    global MODE, STEPS_LEFT
    global PREV_FRONT, PREV_BACK, TREND_FRONT, TREND_BACK

    front = _s(state, "front", 1000.0)
    back  = _s(state, "back", 1000.0)

    # update trends (are they getting closer or farther?)
    if PREV_FRONT is not None:
        if front < PREV_FRONT - TREND_EPS:
            TREND_FRONT += 1 # getting closer
        elif front > PREV_FRONT + TREND_EPS:
            TREND_FRONT = 0 # moving away -> cancel trend
        # else: within noise -> keep current TREND_FRONT
    else:
        TREND_FRONT = 0
    PREV_FRONT = front

    if PREV_BACK is not None:
        if back < PREV_BACK - TREND_EPS:
            TREND_BACK += 1
        elif back > PREV_BACK + TREND_EPS:
            TREND_BACK = 0
    else:
        TREND_BACK = 0
    PREV_BACK = back

    # Only trigger if inside threshold AND trending closer
    trigger = None  # "FRONT" | "BACK" | None
    if front < BLOCK_THR and TREND_FRONT >= TREND_MIN:
        trigger = "FRONT"
    elif back < BLOCK_THR and TREND_BACK >= TREND_MIN:
        trigger = "BACK"

    if not trigger:
        return

    if trigger == "FRONT":
        left_clear  = _min_of(state, LEFT_FWD)
        right_clear = _min_of(state, RIGHT_FWD)
    else:
        left_clear  = _min_of(state, LEFT_BACK)
        right_clear = _min_of(state, RIGHT_BACK)

    # Pick the clearer side
    if right_clear >= left_clear:
        MODE = "RIGHT_A"
    else:
        MODE = "LEFT_A"
    STEPS_LEFT = N_SWITCH

def _step_lane_action():
    """Emit one steering action for the lane switch FSM and advance it."""
    global MODE, STEPS_LEFT
    if MODE == "IDLE":
        return "NOTHING"

    if MODE == "RIGHT_A":
        act = "STEER_RIGHT"
    elif MODE == "RIGHT_B":
        act = "STEER_LEFT"
    elif MODE == "LEFT_A":
        act = "STEER_LEFT"
    elif MODE == "LEFT_B":
        act = "STEER_RIGHT"
    else:
        act = "NOTHING"

    STEPS_LEFT -= 1
    if STEPS_LEFT <= 0:
        if MODE == "RIGHT_A":
            MODE, STEPS_LEFT = "RIGHT_B", N_SWITCH
        elif MODE == "LEFT_A":
            MODE, STEPS_LEFT = "LEFT_B", N_SWITCH
        else:
            MODE, STEPS_LEFT = "IDLE", 0
    return act

def _speed_action(state):
    vx = float((state.get("velocity") or {}).get("x", 0.0) or 0.0)
    vy = float((state.get("velocity") or {}).get("y", 0.0) or 0.0)
    print("vx", vx)
    print("vy", vy)

    if math.isclose(vx, TARGET_VX, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return "NOTHING"

    if vx < TARGET_VX - VX_BAND:
        return "ACCELERATE"
    if vx > TARGET_VX + VX_BAND:
        return "DECELERATE"
    return "NOTHING"

LAST_TICK = None  # track elapsed_ticks across calls

def _reset_state():
    global MODE, STEPS_LEFT
    global PREV_FRONT, PREV_BACK, TREND_FRONT, TREND_BACK
    global LAST_TICK

    MODE = "IDLE"
    STEPS_LEFT = 0
    PREV_FRONT = None
    PREV_BACK = None
    TREND_FRONT = 0
    TREND_BACK = 0
    LAST_TICK = None


def return_action(state: dict):
    global LAST_TICK
    t = int((state.get("elapsed_ticks") or 0))
    did_crash = bool(state.get("did_crash", False))

    # Reset variables
    if did_crash or t == 0 or (LAST_TICK is not None and t < LAST_TICK):
        _reset_state()
    LAST_TICK = t

    # --- NEW: ramp TARGET_VX gently with ticks (bounded) ---
    global TARGET_VX, CURRENT_TARGET_VX
    base_target = BASE_TARGET_VX + RAMP_PER_TICK * t
    target = min(MAX_TARGET_VX, base_target)

    # Optional adaptive nudge: if something is actually closing, bias a bit.
    # (TREND_* already updated in _maybe_start_switch)
    try:
        if TREND_FRONT >= TREND_MIN and TREND_BACK == 0:
            target -= 0.4  # ease off if front is closing
        elif TREND_BACK >= TREND_MIN and TREND_FRONT == 0:
            target += 0.4  # speed up a touch if back is closing
    except NameError:
        pass

    # Clamp and apply
    CURRENT_TARGET_VX = max(BASE_TARGET_VX, min(MAX_TARGET_VX, target))
    TARGET_VX = CURRENT_TARGET_VX
    # --- end NEW ---

    global MODE

    # If not already switching lanes, check sensors to possibly start
    if MODE == "IDLE":
        _maybe_start_switch(state)

    actions = []
    for _ in range(BATCH_SIZE):
        if MODE != "IDLE":
            actions.append(_step_lane_action())
        else:
            actions.append(_speed_action(state))
    return actions





# BATCH_LEN = 12
# FRONT_CLEAR = 999.0   # px (tweak)
# SIDE_CLEAR  = 250.0   # px (tweak)
#
# def return_action(state: dict):
#     sensors = state.get("sensors", {}) or {}
#
#     # print("front:", state["sensors"].get("front"))
#     # state.get("tick", 0)
#     # if state.get("tick") == 0:
#     #     print("Keys:", list(state.keys()))
#     #     print("Velocity keys:", list(state.get("velocity", {}).keys()))
#     #     print("Sensor names:", list(state.get("sensors", {}).keys()))
#
#     # t = state.get("tick", 0)
#     # if t % 20 == 0:
#     #     v = state.get("velocity", {})
#     #     sensors = state.get("sensors", {})
#     #     speed = (v.get("x", 0.0) ** 2 + v.get("y", 0.0) ** 2) ** 0.5
#     #     print(
#     #         f"t={t:4d}  dist={state.get('distance',0):.1f}  "
#     #         f"v=({v.get('x',0):.1f},{v.get('y',0):.1f})  |speed|={speed:.1f}  "
#     #         f"front={sensors.get('front'):.1f}  lf={sensors.get('left_front'):.1f}  "
#     #         f"rf={sensors.get('right_front'):.1f}  left={sensors.get('left_side'):.1f}  "
#     #         f"right={sensors.get('right_side'):.1f}"
#     #     )
#
#     # Print the sensor data
#     # print("Sensors:", sensors)
#
#     # Get the sensor values
#     # Sets the sensors to 1000 as default
#     left_side = float(sensors.get("left_side", 1000))
#     # left_side_front = float(sensors.get("left_side_front", 1000))
#     left_front = float(sensors.get("left_front", 1000))
#     # front_left_front = float(sensors.get("front_left_front", 1000))
#     front = float(sensors.get("front", 1000))
#     # front_right_front = float(sensors.get("front_right_front", 1000))
#     right_front = float(sensors.get("right_front", 1000))
#     # right_side_front = float(sensors.get("right_side_front", 1000))
#     right_side = float(sensors.get("right_side", 1000))
#     # right_side_back = float(sensors.get("right_side_back", 1000))
#     # right_back = float(sensors.get("right_back", 1000))
#     # back_right_back = float(sensors.get("back_right_back", 1000))
#     back = float(sensors.get("back", 1000))
#     # back_left_back = float(sensors.get("back_left_back", 1000))
#     # left_back = float(sensors.get("left_back", 1000))
#     # left_side_back = float(sensors.get("left_side_back", 1000))
#
#     #TODO: Calculate the speed from the velocity
#
#     velocity = state.get("velocity", {})
#
#     velocity_x = float(velocity.get("x", 0.0))
#     print("Velocity_x: ", velocity_x)
#     velocity_y = float(velocity.get("y", 0.0))
#     print("Velocity_y: ", velocity_y)
#     # speed = (velocity_x ** 2 + velocity_y ** 2) ** 0.5
#     # speed = (v.get("x", 0.0) ** 2 + v.get("y", 0.0) ** 2) ** 0.5
#
#
#     # print("Front:", front)
#     # print("Left side:", left_side)
#
#
#     # TODO: Change all this for a better algorithm
#     # TODO: If speed is above x stay at that speed - A speed that allows maneuvering
#     # --- tiny rule-based policy (baseline) ---
#     # 1) If front is blocked, steer towards the more open side
#     if front < FRONT_CLEAR:
#         # Compare “front-ish” space
#         lf = min(left_front, left_side)
#         rf = min(right_front, right_side)
#         action = "STEER_LEFT" if lf > rf else "STEER_RIGHT"
#
#     # 2) If we’re very close to one wall (side), steer away to stay centered
#     elif left_side < SIDE_CLEAR:
#         action = "STEER_RIGHT"
#     elif right_side < SIDE_CLEAR:
#         action = "STEER_LEFT"
#
#     # 3) Otherwise keep speed up
#     elif velocity_x < 10:
#         action = "ACCELERATE"
#
#     else:
#         action = "NOTHING"
#
#     # optional: gentle braking if something’s too close behind to avoid weird rebounds
#     if back < 400:
#         action = "DECELERATE"
#
#     # TODO: CHANGE IT TO SEND NOT ONLY THE SAME ACTION (Maybe)
#     # Auto-batch the single decision
#     return [action] * BATCH_LEN


# --------- anti-snake, center-seeking controller with phases ---------

# Distances
# FRONT_STOP      = 120.0     # emergency stop distance
# CAUTIOUS_FRONT  = 250.0     # start avoidance if front < this
# SIDE_ALERT      = 180.0     # side is "near" here
# SIDE_WALL       = 110.0     # too close to wall -> immediate steer away
#
# # Speeds (forward vx)
# VX_FAST         = 12.0      # cruise when clear/straight
# VX_TURN         =  7.0      # while turning/avoiding
# SPEED_BAND      =  1.0      # deadband to avoid accel/decel spam
#
# # Lateral control
# VY_TOL          = 0.6       # when |vy| below this, we consider drift "small"
# VY_STRONG       = 1.4       # strong damping above this
# CENTER_TOL      = 0.06      # normalized center error deadband
# CENTER_STRONG   = 0.12      # stronger centering threshold
#
# # Phase timing (ticks)
# AVOID_TICKS     = 8         # how long to commit to a dodge
# RECENTER_TICKS  = 10        # gentle recenter after dodge
#
# BATCH_LEN       = 5
#
# # Tiny controller memory
# _STATE = {"phase": "cruise", "ticks": 0}  # phases: cruise, avoid_left, avoid_right, recenter
#
# def _g(d):  # safe get
#     return float(1000 if d in (None, "") else d)
#
# def _set_phase(p, t):
#     _STATE["phase"], _STATE["ticks"] = p, t
#
# def return_action(state: dict):
#     s = state.get("sensors", {}) or {}
#     v = state.get("velocity", {}) or {}
#     print(v)
#
#     # Sensors
#     left_side   = _g(s.get("left_side"))
#     right_side  = _g(s.get("right_side"))
#     left_front  = _g(s.get("left_front"))
#     right_front = _g(s.get("right_front"))
#     front       = _g(s.get("front"))
#     back        = _g(s.get("back"))
#
#     # Velocities
#     vx = float(v.get("x", 0.0))   # forward
#     vy = float(v.get("y", 0.0))   # lateral (+ left / - right; flip LEFT/RIGHT below if opposite in your sim)
#
#     # Helpers
#     denom = max(1.0, left_side + right_side)
#     center_err = (right_side - left_side) / denom  # + => more room right (we’re closer to left)
#     open_left  = 0.6 * left_front  + 0.4 * left_side
#     open_right = 0.6 * right_front + 0.4 * right_side
#
#     action = "NOTHING"
#
#     # -------- SAFETIES --------
#     if back < 100.0:
#         action = "DECELERATE"
#     elif front < FRONT_STOP:
#         action = "DECELERATE"
#     # Hard side guards (never let it scrape walls)
#     elif left_side < SIDE_WALL:
#         action = "STEER_RIGHT"
#         _set_phase("recenter", RECENTER_TICKS)
#     elif right_side < SIDE_WALL:
#         action = "STEER_LEFT"
#         _set_phase("recenter", RECENTER_TICKS)
#
#     else:
#         # -------- PHASE MANAGEMENT --------
#         if _STATE["phase"] == "cruise":
#             # If front is tight, pick the more open side to avoid
#             if front < CAUTIOUS_FRONT:
#                 if open_left > open_right:
#                     _set_phase("avoid_left", AVOID_TICKS)
#                 elif open_right > open_left:
#                     _set_phase("avoid_right", AVOID_TICKS)
#             # Also, if a side is getting near, bias away briefly
#             elif left_side < SIDE_ALERT and right_side >= SIDE_ALERT:
#                 _set_phase("avoid_right", AVOID_TICKS // 2)
#             elif right_side < SIDE_ALERT and left_side >= SIDE_ALERT:
#                 _set_phase("avoid_left", AVOID_TICKS // 2)
#
#         # Decide action based on phase
#         phase = _STATE["phase"]
#         ticks = _STATE["ticks"]
#
#         if phase == "avoid_left":
#             # Commit to LEFT steer, but also damp lateral drift
#             action = "STEER_LEFT"
#             _STATE["ticks"] = max(0, ticks - 1)
#             # when drift is small and front is no longer tight, go recenter
#             if _STATE["ticks"] == 0 or (abs(vy) <= VY_TOL and front >= CAUTIOUS_FRONT):
#                 _set_phase("recenter", RECENTER_TICKS)
#
#         elif phase == "avoid_right":
#             action = "STEER_RIGHT"
#             _STATE["ticks"] = max(0, ticks - 1)
#             if _STATE["ticks"] == 0 or (abs(vy) <= VY_TOL and front >= CAUTIOUS_FRONT):
#                 _set_phase("recenter", RECENTER_TICKS)
#
#         elif phase == "recenter":
#             _STATE["ticks"] = max(0, ticks - 1)
#             # First priority in recenter: kill lateral drift
#             if abs(vy) > VY_TOL:
#                 # If vy > 0 drifting left -> steer RIGHT (oppose drift)
#                 action = "STEER_RIGHT" if vy > 0 else "STEER_LEFT"
#             else:
#                 # Gentle center-seeking when drift is small
#                 if abs(center_err) > CENTER_STRONG:
#                     action = "STEER_RIGHT" if center_err > 0 else "STEER_LEFT"
#                 elif abs(center_err) > CENTER_TOL:
#                     action = "STEER_RIGHT" if center_err > 0 else "STEER_LEFT"
#                 else:
#                     action = "NOTHING"
#
#             if _STATE["ticks"] == 0 and abs(vy) <= VY_TOL and abs(center_err) <= CENTER_TOL:
#                 _set_phase("cruise", 0)
#
#         else:  # cruise (default)
#             # If nothing urgent, prefer to damp drift lightly, then center
#             if abs(vy) > VY_STRONG:
#                 action = "STEER_RIGHT" if vy > 0 else "STEER_LEFT"
#             elif abs(center_err) > CENTER_STRONG:
#                 action = "STEER_RIGHT" if center_err > 0 else "STEER_LEFT"
#             elif front < CAUTIOUS_FRONT:
#                 # front tight but no phase set (rare), nudge to the open side
#                 if open_left > open_right:
#                     action = "STEER_LEFT"
#                 elif open_right > open_left:
#                     action = "STEER_RIGHT"
#                 else:
#                     action = "NOTHING"
#             else:
#                 action = "NOTHING"
#
#     # -------- FORWARD SPEED CONTROL --------
#     turning_or_tight = action in ("STEER_LEFT", "STEER_RIGHT") or front < CAUTIOUS_FRONT or _STATE["phase"] in ("avoid_left", "avoid_right")
#     target_vx = VX_TURN if turning_or_tight else VX_FAST
#
#     if action == "NOTHING":
#         if vx < target_vx - SPEED_BAND:
#             action = "ACCELERATE"
#         elif vx > target_vx + SPEED_BAND:
#             action = "DECELERATE"
#         # else keep NOTHING
#
#     return [action] * BATCH_LEN




if __name__ == '__main__':
    import pygame
    from src.game.core import initialize_game_state, game_loop
    seed_value = None
    pygame.init()
    initialize_game_state("http://localhost:9052/predict", seed_value)
    game_loop(verbose=True) # For pygame window
    pygame.quit()