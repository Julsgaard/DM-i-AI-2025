from racecar_env import RaceCarEnv
e = RaceCarEnv()
obs, info = e.reset()
obs2, r, term, trunc, info = e.step(e.action_space.sample())
print(obs.shape, r, term, trunc)
