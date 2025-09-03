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
# TODO: First priority - TEST THIS WITH SEED 5 AND 6 - Use more sensors for example the front_left_front and front_right_front.
#  These can determine if there is a car coming in the other lane. left_side_back, left_side_front, left_side,
#  right_side_front, right_side, right_side_back are also important to check if there is a car there.
#  It should not navigate into one of these if there is a car and if there is no option then deaccelarate if the car is
#  coming in front and accelerate if it is coming from behind (CAN BE TESTED WITH SEED 4, 7 AND 8).
# TODO: First Priority - SEED 3 - Almost impossible seed
# TODO: Second priority - It accelerates the same amount as the batch size and then it does nothing for the same amount as the batch size.
#  This might create problems.
# TODO: Second priority - It needs to decide whether it should deaccelerate or steer left or right
# TODO: Second priority - It should try to stay in the middle lane. Or atleast not be in the lanes at the edge/wall of the map

# Tiny state (persists across calls)
mode = "IDLE" # IDLE | RIGHT_A | RIGHT_B | LEFT_A | LEFT_B
steps_left = 0

# Remember last distances + trend counters
prev_front = None
prev_back = None
trend_front = 0
trend_back = 0
# Trend sensitivity
trend_eps = 5.0 # pixels change to consider "meaningful"
trend_min = 2 # need this many consecutive "getting closer" ticks to trigger TODO: Can maybe be 1?

batch_size = 6 # how many actions performed per call
n_switch = 48 # steps per half-switch (A or B)
block_thr = 999.0 # obstacle if sensor < this
target_vx = 20 # desired vx - Is updated each call
vx_band = 0.15 # dead zone around target_vx
base_target_vx = 10.05 # start slow
max_target_vx = 999 # The maximum target VX
ramp_per_tick = 0.05 # vx gained per tick
ramp_ticks = 0 # only counts when NOT steering
current_target_vx = base_target_vx  # computed each call

rel_tol = 1e-6
abs_tol = 1e-3

# TODO: ---DONE--- It needs to gradually accelerate infinitely. This is because the cars are spawning at the same speed as the race-car.
#  If i only accelerate it will be too fast to dodge, but if i gradually accelerate it might be enough to dodge.
#  I think that even though you are moving at 100 velocity the cars will spawn at the same speed.
#  This is confirmed by what it says in place_car function in core.py

# Sensor groups
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
    global mode, steps_left
    global prev_front, prev_back, trend_front, trend_back

    front = _sensor(state, "front", 1000.0)
    back  = _sensor(state, "back", 1000.0)

    # update trends (are they getting closer or farther?)
    if prev_front is not None:
        if front < prev_front - trend_eps:
            trend_front += 1 # getting closer
        elif front > prev_front + trend_eps:
            trend_front = 0 # moving away -> cancel trend
        # else: within noise -> keep current TREND_FRONT
    else:
        trend_front = 0
    prev_front = front

    if prev_back is not None:
        if back < prev_back - trend_eps:
            trend_back += 1
        elif back > prev_back + trend_eps:
            trend_back = 0
    else:
        trend_back = 0
    prev_back = back

    # Only trigger if inside threshold AND trending closer
    trigger = None  # "FRONT" | "BACK" | None
    if front < block_thr and trend_front >= trend_min:
        trigger = "FRONT"
    elif back < block_thr and trend_back >= trend_min:
        trigger = "BACK"

    if not trigger:
        return

    if trigger == "FRONT":
        left_clear = _min_of(state, LEFT_FWD)
        right_clear = _min_of(state, RIGHT_FWD)
    else:
        left_clear = _min_of(state, LEFT_BACK)
        right_clear = _min_of(state, RIGHT_BACK)

    # Pick the clearer side
    if right_clear >= left_clear:
        mode = "RIGHT_A"
    else:
        mode = "LEFT_A"
    steps_left = n_switch

def _step_lane_action():
    """One steering action for the lane switch FSM and advance it."""
    global mode, steps_left
    if mode == "IDLE":
        return "NOTHING"

    if mode == "RIGHT_A":
        act = "STEER_RIGHT"
    elif mode == "RIGHT_B":
        act = "STEER_LEFT"
    elif mode == "LEFT_A":
        act = "STEER_LEFT"
    elif mode == "LEFT_B":
        act = "STEER_RIGHT"
    else:
        act = "NOTHING"

    steps_left -= 1
    if steps_left <= 0:
        if mode == "RIGHT_A":
            mode, steps_left = "RIGHT_B", n_switch
        elif mode == "LEFT_A":
            mode, steps_left = "LEFT_B", n_switch
        else:
            mode, steps_left = "IDLE", 0
    return act

def _speed_action(state):
    # pause accel/decel while steering - This is just a failsafe
    if mode != "IDLE":
        return "NOTHING"

    vx = float((state.get("velocity") or {}).get("x", 0.0) or 0.0)
    print("vx", vx)

    # vy = float((state.get("velocity") or {}).get("y", 0.0) or 0.0)
    # print("vy", vy)

    if math.isclose(vx, target_vx, rel_tol=rel_tol, abs_tol=abs_tol):
        return "NOTHING"

    if vx < target_vx - vx_band:
        return "ACCELERATE"
    if vx > target_vx + vx_band:
        return "DECELERATE"
    return "NOTHING"

last_tick = None

# Reset all state variables to prevent it from remembering anything from previous runs
def _reset_state():
    global mode, steps_left
    global prev_front, prev_back, trend_front, trend_back
    global last_tick
    global ramp_ticks

    ramp_ticks = 0
    mode = "IDLE"
    steps_left = 0
    prev_front = None
    prev_back = None
    trend_front = 0
    trend_back = 0
    last_tick = None


def return_action(state: dict):
    global last_tick, mode, target_vx, current_target_vx, ramp_ticks

    t = int((state.get("elapsed_ticks") or 0)) # Tick count from the game
    did_crash = bool(state.get("did_crash", False)) # Crash status from the game

    # Reset variables
    if did_crash or t == 0 or (last_tick is not None and t < last_tick):
        _reset_state() # reset all state variables
        print("RESET")

    # compute dt using the previous last_tick. I had to create a separate tick to prevent ramping the acceleration
    # too much after steering
    prev_t = last_tick
    dt = 0 if prev_t is None else max(0, t - prev_t)

    if mode == "IDLE":
        # Only count idle time. Steering time does not increase the ramp
        ramp_ticks += dt
    # Update last_tick for the next call
    last_tick = t

    # Calculate the target velocity
    base_target = base_target_vx + ramp_per_tick * ramp_ticks
    target = min(max_target_vx, base_target)

    # Adaptive nudge
    try:
        if trend_front >= trend_min and trend_back == 0:
            target -= 0.4
        elif trend_back >= trend_min and trend_front == 0:
            target += 0.4
    except NameError:
        pass

    # Clamp and apply
    current_target_vx = max(base_target_vx, min(max_target_vx, target))
    target_vx = current_target_vx

    # If not already switching lanes, check sensors to possibly start
    if mode == "IDLE":
        _maybe_start_switch(state)

    actions = []
    for _ in range(batch_size):
        if mode != "IDLE":
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