import argparse
import functools
import os
import random
import time

import retro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Emerald-GBAdvance")
    parser.add_argument("--frameskip", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()

def load_pokemon_game(env: str):
    gym_env = retro.make(env, inttype=retro.data.Integrations.ALL, render_mode="rgb_array")

def benchmark_function(func, iterations=1000, warmup=10):
    print(f"Warming up for {warmup} iterations...")
    for _ in range(warmup):
        func()

    print(f"Benchmarking for {iterations} iterations...")
    start_time = time.time()
    for _ in range(iterations):
        func()
    end_time = time.time()

    it_per_sec = iterations / (end_time - start_time)
    return it_per_sec

def random_env_step(env):
    env.step(env.action_space.sample())


def main(args):
    retro.data.Integrations.add_custom_path(SCRIPT_DIR)
    env = retro.make(args.env, inttype=retro.data.Integrations.ALL, render_mode="rgb_array")

    env.reset()
    steps_per_sec = benchmark_function(env.em.step, args.iterations)
    print(f"FPS: {steps_per_sec}")
    print("---")

    env.reset()
    steps_per_sec = benchmark_function(functools.partial(random_env_step, env), args.iterations)
    print(f"Env. it/s: {steps_per_sec}")
    print("---")

if __name__ == "__main__":
    args = parse_args()
    main(args)
