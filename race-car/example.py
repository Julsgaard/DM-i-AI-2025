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

# State variables for lane switching
mode = "IDLE" # IDLE | RIGHT_A | RIGHT_B | LEFT_A | LEFT_B
steps_left = 0 # Steps left in current mode

# Remember last distances + trend counters
prev_front = None
prev_back = None
trend_front = 0
trend_back = 0
# Trend sensitivity
trend_eps = 5.0 # Pixels change to consider "meaningful"
trend_min = 1 # Need this many consecutive "getting closer" ticks to trigger TODO: Can maybe be 1?

batch_size = 6 # How many actions performed per call
n_switch = 48 # Steps per half-switch (A or B)
block_thr = 999.0 # Obstacle if sensor < this
target_vx = 20 # Desired vx - Is updated each call
vx = 10 # The velocity x from the sensor - Only used for printing
vx_band = 0.15 # Dead zone around target_vx
base_target_vx = 10.05 # Start slow
max_target_vx = 999 # The maximum target VX
ramp_per_tick = 0.0525 # vx gained per tick
ramp_ticks = 0 # Only counts when NOT steering
current_target_vx = base_target_vx  # Computed each call

# For when sides are blocked
side_clear_thr = 350.0 # If side sensors < this, consider it blocked
front_safe_thr = 850.0 # How far the forward-diagonal must be to enter that lane
side_gap_thr   = 325.0 # Min clearance beside you in target lane to be safe
pending_escape = False # True if we need to escape a blocked side

rel_tol = 1e-6 # For math.isclose
abs_tol = 1e-3 # For math.isclose

speedup_tick = 500 # After this tick the target speed will increase faster
speedup_multiplier = 1.45 # The added to the speedup per tick

# TODO: ---DONE--- It needs to gradually accelerate infinitely. This is because the cars are spawning at the same speed as the race-car.
#  If i only accelerate it will be too fast to dodge, but if i gradually accelerate it might be enough to dodge.
#  I think that even though you are moving at 100 velocity the cars will spawn at the same speed.
#  This is confirmed by what it says in place_car function in core.py

# Sensor groups
left_fwd   = ("left_side_front", "left_front", "front_left_front")
right_fwd  = ("front_right_front", "right_front", "right_side_front")
right_back = ("right_side_back", "right_back", "back_right_back")
left_back  = ("back_left_back", "left_back", "left_side_back")

LEFT_SIDE_GROUP  = ("left_side_front", "left_side", "left_side_back")
RIGHT_SIDE_GROUP = ("right_side_front", "right_side", "right_side_back")

def _sensor(state, name, default=1000.0):
    return float((state.get("sensors") or {}).get(name, default) or default)

def _min_of(state, names):
    sens = state.get("sensors") or {}
    vals = [float(sens.get(n, 1000.0) or 1000.0) for n in names if n in sens]
    return min(vals) if vals else 1000.0

def _maybe_start_switch(state):
    """
    Trigger a lane switch if front or back is blocked.
    Choose clearer side: compare left vs right groups (front case uses FWD groups, back case uses BACK groups).
    """
    global mode, steps_left, prev_front, prev_back, trend_front, trend_back, pending_escape

    front = _sensor(state, "front", 1000.0)
    back  = _sensor(state, "back", 1000.0)

    # Update trends - Are the cars getting closer or further away?
    if prev_front is not None:
        if front < prev_front - trend_eps:
            trend_front += 1 # Getting closer
        elif front > prev_front + trend_eps:
            trend_front = 0 # Moving away - cancel trend
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

    # TODO: MAKE SURE THIS IS CORRECT!!!!
    # Forward-diagonals (primary look-ahead per side)
    flf = _sensor(state, "front_left_front", 1000.0)
    frf = _sensor(state, "front_right_front", 1000.0)

    # Rear clearance per side
    left_back_min = _min_of(state, left_back)
    right_back_min = _min_of(state, right_back)

    # Side-adjacent gaps (cars alongside you in that lane)
    left_side_min = _min_of(state, LEFT_SIDE_GROUP)
    right_side_min = _min_of(state, RIGHT_SIDE_GROUP)

    # A side is enterable only if ALL three checks pass:
    # (1) forward-diagonal, (2) rear gap, (3) side-adjacent gap
    left_enterable = (flf >= front_safe_thr) and (left_back_min >= side_clear_thr) and (left_side_min >= side_gap_thr)
    right_enterable = (frf >= front_safe_thr) and (right_back_min >= side_clear_thr) and (right_side_min >= side_gap_thr)

    # Boxed in? -> stay in escape mode do not start a turn yet
    if not left_enterable and not right_enterable:
        pending_escape = True
        return
    else:
        pending_escape = False

    # Choose side - prefer any enterable side; if both, pick the one with the larger bottleneck margin
    # Bottleneck = the tightest of the three constraints
    left_bottleneck = min(flf, left_back_min, left_side_min)
    right_bottleneck = min(frf, right_back_min, right_side_min)

    if left_enterable and not right_enterable:
        mode = "LEFT_A"
    elif right_enterable and not left_enterable:
        mode = "RIGHT_A"
    else:
        mode = "RIGHT_A" if right_bottleneck >= left_bottleneck else "LEFT_A"

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
    global vx

    # Pause accelerate/decelerate while steering - This is just a failsafe
    if mode != "IDLE":
        return "NOTHING"

    vx = float((state.get("velocity") or {}).get("x", 0.0) or 0.0)

    # vy = float((state.get("velocity") or {}).get("y", 0.0) or 0.0)
    # print("vy", vy)

    # Escape behavior
    if pending_escape:
        front = _sensor(state, "front", 1000.0)
        back = _sensor(state, "back", 1000.0)
        # If front is the threat (or both) slow down. If back is the threat speed up a bit
        if front < block_thr and (back >= block_thr or front <= back):
            print("ESCAPE DECELERATE")
            return "DECELERATE"
        if back < block_thr and front >= block_thr:
            print("ESCAPE ACCELERATE")
            return "ACCELERATE"
        print("ESCAPE DEFAULT")
        # Else prefer decelerate
        return "ACCELERATE"

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
    global mode, steps_left, prev_front, prev_back, trend_front, trend_back, last_tick, ramp_ticks, pending_escape

    ramp_ticks = 0
    mode = "IDLE"
    steps_left = 0
    prev_front = None
    prev_back = None
    trend_front = 0
    trend_back = 0
    last_tick = None
    pending_escape = False

def return_action(state: dict):
    global last_tick, mode, target_vx, current_target_vx, ramp_ticks

    print("Velocity X: ", vx)

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
    if t < speedup_tick:
        base_target = base_target_vx + ramp_per_tick * ramp_ticks
        # print("old", base_target)
    else:
        base_target = base_target_vx + ramp_per_tick * speedup_multiplier * ramp_ticks
        # print("new", base_target)
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
    seed_value = None
    pygame.init()
    initialize_game_state("http://localhost:9052/predict", seed_value)
    game_loop(verbose=True) # For pygame window
    pygame.quit()