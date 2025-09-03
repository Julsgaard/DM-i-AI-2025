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


# -------------------- LANE SWITCHER ---------------------------
# TODO: First priority - At the moment the race-car accelerates more after having turned, as it is trying to catch up
#  because it was not able to accelerate while turning. This creates problems because cars will then be spawning faster
#  than usual because of this sudden acceleration.
# TODO: First priority - Use more sensors for example the front_left_front and front_right_front.
#  These can determine if there is a car coming in the other lane. left_side_back, left_side_front, left_side,
#  right_side_front, right_side, right_side_back are also important to check if there is a car there.
#  It should not navigate into one of these if there is a car and if there is no option then deaccelarate if the car is
#  coming in front and accelerate if it is coming from behind.
# TODO: Second priority - It accelerates the same amount as the batch size and then it does nothing for the same amount as the batch size.
#  This might create problems.
# TODO: Second priority - It needs to decide whether it should deaccelerate or steer left or right
# TODO: Second priority - It should try to stay in the middle lane. Or atleast not be in the lanes at the edge/wall of the map
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
BATCH_SIZE = 6
N_SWITCH = 48
BLOCK_THR = 999.0 # obstacle if sensor < this
TARGET_VX = 20
VX_BAND = 0.15
BASE_TARGET_VX = 10.05 # start slow
MAX_TARGET_VX = 999 # The maximum target VX
RAMP_PER_TICK = 0.05 # vx gained per tick
RAMP_TICKS = 0 # only counts when NOT steering
# pause_acceleration = False # We need to pause the acceleration while steering or else it accelerates too quickly after a turn
MAX_TARGET_AHEAD = 0.6 # don't let target exceed current vx by > 0.6
CURRENT_TARGET_VX = BASE_TARGET_VX  # computed each call

REL_TOL = 1e-6
ABS_TOL = 1e-3

# TODO: ---DONE--- It needs to gradually accelerate infinitely. This is because the cars are spawning at the same speed as the race-car.
#  If i only accelerate it will be too fast to dodge, but if i gradually accelerate it might be enough to dodge.
#  I think that even though you are moving at 100 velocity the cars will spawn at the same speed.
#  This is confirmed by what it says in place_car function in core.py

# Sensor groups (use the names you already have in your sim payload)
LEFT_FWD   = ("front_left_front", "left_side_front")
RIGHT_FWD  = ("front_right_front", "right_side_front")
LEFT_BACK  = ("back_left_back", "left_side_back")
RIGHT_BACK = ("back_right_back", "right_side_back")


def _sensor(state, name, default=1000.0):
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

    front = _sensor(state, "front", 1000.0)
    back  = _sensor(state, "back", 1000.0)

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
    print("vx", vx)

    # vy = float((state.get("velocity") or {}).get("y", 0.0) or 0.0)
    # print("vy", vy)

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
    global LAST_TICK, MODE, TARGET_VX, CURRENT_TARGET_VX
    t = int((state.get("elapsed_ticks") or 0))
    did_crash = bool(state.get("did_crash", False))

    # Reset variables
    if did_crash or t == 0 or (LAST_TICK is not None and t < LAST_TICK):
        _reset_state()
    LAST_TICK = t

    # ramp TARGET_VX gently with ticks (bounded)
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


if __name__ == '__main__':
    import pygame
    from src.game.core import initialize_game_state, game_loop
    seed_value = 2
    pygame.init()
    initialize_game_state("http://localhost:9052/predict", seed_value)
    game_loop(verbose=True) # For pygame window
    pygame.quit()