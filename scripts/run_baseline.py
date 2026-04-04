import os
import json
from openai import OpenAI
from typing import Any
from tasks.easy import make_easy_env, grade as grade_easy, description as desc_easy
from tasks.medium import make_medium_env, grade as grade_medium, description as desc_medium
from tasks.hard import make_hard_env, grade as grade_hard, description as desc_hard
from env.models import Observation, Action

api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "OPENAI_API_KEY must be set in environment!"
client = OpenAI(api_key=api_key)

TASKS = [
    ("Easy", make_easy_env, grade_easy, desc_easy),
    ("Medium", make_medium_env, grade_medium, desc_medium),
    ("Hard", make_hard_env, grade_hard, desc_hard)
]


def agent_act(observation: Observation) -> Action:
    prompt = f"""
You are an ER triage AI. Given the following observation (JSON), return a single action (JSON) to maximize patient outcomes.

Observation:
{json.dumps(observation.dict(), indent=2)}

Action schema: assign_priority, reassign_priority, allocate_resource, escalate_patient, wait.
Respond with ONLY a valid JSON action, nothing else.
"""
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are a medical triage AI. Return only valid JSON."},
                  {"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    action_json = response.choices[0].message.content
    try:
        action_dict = json.loads(action_json)
        if 'type' not in action_dict:
            action_dict['type'] = 'wait'
        return Action.parse_obj(action_dict)
    except Exception as e:
        print(f"Agent output error: {e}, got: {action_json}")
        return Action.parse_obj({'type': 'wait'})

def run_task(make_env, grade_fn, desc, seed):
    env = make_env(seed=seed)
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        action = agent_act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1
    score = grade_fn(env)
    return total_reward, score, steps

def main():
    print("=== Dynamic ER Triage Simulation Baseline ===\n")
    for name, make_env, grade_fn, desc in TASKS:
        print(f"--- {name} Task ---")
        print(desc)
        total_reward, score, steps = run_task(make_env, grade_fn, desc, seed=42)
        print(f"Steps: {steps}, Total Reward: {total_reward:.2f}, Grader Score: {score:.3f}\n")

if __name__ == "__main__":
    main()
