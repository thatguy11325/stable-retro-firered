import os
import time

import retro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
  retro.data.Integrations.add_custom_path(SCRIPT_DIR)
  print("Emerald exists: ", "Emerald-GBAdvance" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
  env = retro.make("Emerald-GBAdvance", inttype=retro.data.Integrations.ALL, render_mode="human")
  env.reset()

  n_steps = 1000

  start = time.time()
  for _ in range(n_steps):
    x = env.step(env.action_space.sample())
    env.render()
  env.close()
  end = time.time()

  print(f"Stable Retro ran {n_steps} in {end-start} seconds or {n_steps / (end-start)} steps/second")

if __name__ == "__main__":
  main()
