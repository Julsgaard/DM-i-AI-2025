# api_rl.py
import time, datetime, os
from fastapi import FastAPI, Body
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from stable_baselines3 import PPO
import numpy as np

DEBUG = True
_debug_counter = 0

def _min_front_block(s):
    keys = ["front","front_left_front","front_right_front","left_front","right_front"]
    return min(float(s.get(k, 1000.0) or 1000.0) for k in keys)

def _min_back_block(s):
    keys = ["back","back_left_back","back_right_back","left_back","right_back"]
    return min(float(s.get(k, 1000.0) or 1000.0) for k in keys)

HOST = "0.0.0.0"
PORT = 9052
BATCH_LEN = 10  # small, keeps latency low

SENSOR_ORDER = [
    "left_side","left_side_front","left_front","front_left_front",
    "front","front_right_front","right_front","right_side_front",
    "right_side","right_side_back","right_back","back_right_back",
    "back","back_left_back","left_back","left_side_back",
]
ACTIONS = ["ACCELERATE","DECELERATE","STEER_LEFT","STEER_RIGHT","NOTHING"]

# Try loading the model; if missing, fall back to a trivial policy
MODEL_PATH = "models/ppo_racecar.zip"
RL_MODEL = PPO.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
if RL_MODEL:
    print(f"[API] Loaded RL model: {MODEL_PATH}")
else:
    print(f"[API] RL model not found at {MODEL_PATH}; using fallback policy.")

def _obs_from_request(req: dict) -> np.ndarray:
    sensors = req.get("sensors", {}) or {}
    arr = []
    for k in SENSOR_ORDER:
        v = sensors.get(k, 1000.0)
        v = 1000.0 if v in (None, "") else float(v)
        arr.append(v / 1000.0)
    vel = req.get("velocity", {}) or {}
    vx = float(vel.get("x", 0.0))
    vy = float(vel.get("y", 0.0))
    vx_n = np.clip(vx / 20.0, 0.0, 1.0)
    vy_n = np.clip((vy + 5.0) / 10.0, 0.0, 1.0)
    arr += [vx_n, vy_n]
    return np.array(arr, dtype=np.float32)

def _fallback_policy(req: dict) -> str:
    # minimal safe fallback: if front is blocked, steer away from the shorter side; else accelerate
    s = req.get("sensors", {}) or {}
    front = float(s.get("front", 1000.0) or 1000.0)
    left = float(s.get("left_side", 1000.0) or 1000.0)
    right = float(s.get("right_side", 1000.0) or 1000.0)
    if front < 150.0:
        return "STEER_LEFT" if right < left else "STEER_RIGHT"
    return "ACCELERATE"

def _safety_override(req: dict, proposed: str) -> str:
    s = req.get("sensors", {}) or {}
    left  = float(s.get("left_side", 1000.0) or 1000.0)
    right = float(s.get("right_side", 1000.0) or 1000.0)
    min_front = _min_front_block(s)

    # Hard emergency brake if extremely close
    if min_front < 70.0:
        return "DECELERATE"

    # If close, prefer steering away from the nearer side; light brake if still fast
    if min_front < 200.0:
        steer = "STEER_RIGHT" if left < right else "STEER_LEFT"
        # Prefer to steer; if the model already chose a steer, keep it
        if proposed not in ("STEER_LEFT", "STEER_RIGHT"):
            return steer
        return proposed

    # Otherwise, accept the model's choice
    return proposed

def _decide(req: dict) -> str:
    if USE_REFLEX:
        return _reflex_decide(req)

    # otherwise: RL + safety net
    if RL_MODEL is None:
        proposed = _fallback_policy(req)
    else:
        obs = _obs_from_request(req)
        action_idx, _ = RL_MODEL.predict(obs, deterministic=True)
        proposed = ACTIONS[int(action_idx)]
    # safety (in case you switch RL back on)
    return _safety_override(req, proposed) if 'SAFE_FRONT' in globals() else proposed


# --- Reflex planner thresholds (pixels) ---
COLLISION      = 320.0      # front crash line (already used)
SIDE_COLLISION = 120.0      # left/right crash line from car center
PREPARE_FRONT = 900.0        # start planning when we first see something
COMMIT_FRONT = 650.0         # definitely start moving over
BRAKE_FRONT = 420.0          # never accelerate below this
HARD_FRONT = 340.0           # emergency brake

SIDE_HARD = 140.0  # never steer into a side tighter than this
SIDE_SAFE = 180.0  # prefer not to steer if corridor is tighter than this

PREPARE_BACK = 450.0         # start reacting to rear cars
TAILGATE = 260.0             # rear too close

# State to keep steering consistent over multiple ticks
REFLEX_STATE = {"intent": None, "cooldown": 0}  # intent: "LEFT"|"RIGHT"|None

def _g(s, k, default=1000.0):
    v = s.get(k, default)
    return float(default if v in (None, "") else v)

def _cluster_front(s):
    return min(_g(s,"front"), _g(s,"front_left_front"), _g(s,"front_right_front"),
               _g(s,"left_front"), _g(s,"right_front"))

def _cluster_back(s):
    return min(_g(s,"back"), _g(s,"back_left_back"), _g(s,"back_right_back"),
               _g(s,"left_back"), _g(s,"right_back"))

def _corridor_score_left(s):
    ahead = min(_g(s,"left_front"), _g(s,"front_left_front"))
    side  = _g(s,"left_side")
    rear  = min(_g(s,"left_back"), _g(s,"left_side_back"), _g(s,"back_left_back"))
    base  = 0.6*ahead + 0.3*side + 0.1*rear
    # hard penalty if the side corridor is below the safe margin
    if side < SIDE_SAFE: base -= (500.0 - side)
    return base

def _corridor_score_right(s):
    ahead = min(_g(s,"right_front"), _g(s,"front_right_front"))
    side  = _g(s,"right_side")
    rear  = min(_g(s,"right_back"), _g(s,"right_side_back"), _g(s,"back_right_back"))
    base  = 0.6*ahead + 0.3*side + 0.1*rear
    if side < SIDE_SAFE: base -= (500.0 - side)
    return base


def _side_left_min(s):
    return min(_g(s,"left_side"), _g(s,"left_front"), _g(s,"left_side_back"))

def _side_right_min(s):
    return min(_g(s,"right_side"), _g(s,"right_front"), _g(s,"right_side_back"))


def _best_steer(s):
    left_min  = _side_left_min(s)
    right_min = _side_right_min(s)

    # never steer into a side that is basically unsafe
    if left_min  < SIDE_HARD and right_min >= SIDE_SAFE: return "STEER_RIGHT", "RIGHT"
    if right_min < SIDE_HARD and left_min  >= SIDE_SAFE: return "STEER_LEFT",  "LEFT"
    if left_min  < SIDE_HARD and right_min <  SIDE_HARD: return "DECELERATE", None  # both blocked

    # otherwise prefer the higher-scoring corridor
    left_score  = _corridor_score_left(s)
    right_score = _corridor_score_right(s)
    if left_score >= right_score: return "STEER_LEFT",  "LEFT"
    else:                         return "STEER_RIGHT", "RIGHT"


USE_REFLEX = True  # set False to let pure RL act (still protected by safety)

def _reflex_decide(req: dict) -> str:
    s = req.get("sensors", {}) or {}
    vx = float((req.get("velocity") or {}).get("x", 0.0))

    minF = _cluster_front(s)
    minB = _cluster_back(s)

    # cooldown keeps intent steady for a few ticks so we don't jitter
    if REFLEX_STATE["cooldown"] > 0:
        REFLEX_STATE["cooldown"] -= 1

    # 1) EMERGENCY
    if minF <= HARD_FRONT:
        REFLEX_STATE["intent"] = None
        REFLEX_STATE["cooldown"] = 0
        return "DECELERATE"

    # 2) FRONT danger: prepare/commit a dodge early
    if minF <= PREPARE_FRONT:
        steer, dirn = _best_steer(s)
        # commit once inside COMMIT_FRONT
        if minF <= COMMIT_FRONT:
            REFLEX_STATE["intent"] = dirn
            REFLEX_STATE["cooldown"] = 8  # keep steering this way for a bit
        # never accelerate below BRAKE_FRONT
        if minF <= BRAKE_FRONT and vx > 8.0:
            # if we’re very fast, brake first; otherwise steer
            return "DECELERATE" if vx > 14.0 else steer
        # if we already committed, keep steering that way
        # before returning based on intent:
        if REFLEX_STATE["intent"] == "LEFT" and _side_left_min(s) < SIDE_HARD:  REFLEX_STATE["intent"] = None
        if REFLEX_STATE["intent"] == "RIGHT" and _side_right_min(s) < SIDE_HARD:  REFLEX_STATE["intent"] = None
        return steer

    # 3) REAR danger: don't brake to zero; prefer accelerate or move to space
    if minB <= PREPARE_BACK:
        steer, dirn = _best_steer(s)
        if vx < 6.0:           # being tailgated and slow -> create space
            return "ACCELERATE"
        # if front is clear but one side is much more open, start moving there
        if abs(_corridor_score_left(s) - _corridor_score_right(s)) > 80.0:
            REFLEX_STATE["intent"] = dirn
            REFLEX_STATE["cooldown"] = 6
            return steer
        # otherwise maintain speed
        return "NOTHING"

    # 4) If we set an intent recently, keep it until truly clear
    if REFLEX_STATE["intent"] == "LEFT"  and REFLEX_STATE["cooldown"] > 0: return "STEER_LEFT"
    if REFLEX_STATE["intent"] == "RIGHT" and REFLEX_STATE["cooldown"] > 0: return "STEER_RIGHT"

    # 5) CRUISE: accelerate gently when totally clear
    return "ACCELERATE" if vx < 18.0 else "NOTHING"


app = FastAPI()
_start = time.time()

@app.get("/")
def root():
    return "Your endpoint is running!"

@app.get("/api")
def hello():
    return {"service":"race-car-usecase","uptime":str(datetime.timedelta(seconds=time.time()-_start))}

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    req = request.dict()
    act = _decide(req)

    # ---- DEBUG PRINT (every 10th call) ----
    global _debug_counter
    if DEBUG:
        _debug_counter = (_debug_counter + 1) % 10
        if _debug_counter == 0:
            s = req.get("sensors", {}) or {}
            vx = req.get("velocity", {}).get("x", 0.0)
            vy = req.get("velocity", {}).get("y", 0.0)
            mf = _min_front_block(s)
            mb = _min_back_block(s)
            # show a compact set of sensors
            watch = {k: s.get(k, None) for k in [
                "front", "front_left_front", "front_right_front",
                "left_front", "right_front",
                "left_side", "right_side",
                "left_side_back", "right_side_back",
                "back", "back_left_back", "back_right_back",
                "left_back", "right_back"
            ]}
            lm = _side_left_min(s);
            rm = _side_right_min(s)
            print(
                f"[VAL] vx={vx:.1f} minF={mf:.1f} minB={mb:.1f} lMin={lm:.1f} rMin={rm:.1f} -> {act} | sensors={watch}")

    return RaceCarPredictResponseDto(actions=[act] * BATCH_LEN)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_rl:app", host="0.0.0.0", port=9052, reload=False)
