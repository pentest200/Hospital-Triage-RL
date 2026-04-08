import os
import json
from openai import OpenAI
from typing import Any
from tasks.easy import make_easy_env, grade as grade_easy, description as desc_easy
from tasks.medium import make_medium_env, grade as grade_medium, description as desc_medium
from tasks.hard import make_hard_env, grade as grade_hard, description as desc_hard
from triage_env.models import Observation, Action
from pydantic import TypeAdapter

action_adapter = TypeAdapter(Action)

# Configs - default to OpenAI's gpt-4o-mini
model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
api_base_url = os.environ.get("API_BASE_URL") # if None, hits native OpenAI API

# Fetch key from OPENAI_API_KEY if present, fallback to HF_TOKEN as mandated by hackathon rules
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")

if not api_key:
    raise ValueError("Missing API credentials! Please export OPENAI_API_KEY or HF_TOKEN.")

client_args = {"api_key": api_key}
if api_base_url:
    client_args["base_url"] = api_base_url

client = OpenAI(**client_args)

TASKS = [
    ("Easy", make_easy_env, grade_easy, desc_easy),
    ("Medium", make_medium_env, grade_medium, desc_medium),
    ("Hard", make_hard_env, grade_hard, desc_hard)
]

def agent_act(observation: Observation) -> Action:
    prompt = f"""
You are an ER triage AI. Given the following observation (JSON), return a single action (JSON) to maximize patient outcomes.

Observation:
{json.dumps(observation.model_dump(), indent=2)}

Action schema: assign_priority, reassign_priority, allocate_resource, escalate_patient, wait.
Respond with ONLY a valid JSON action, nothing else.
"""
    kwargs = {
        "model": model_name,
        "messages": [{"role": "system", "content": "You are a medical triage AI. Return only valid JSON."},
                     {"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    
    # Try to enforce json_object response format natively if using OpenAI
    if not api_base_url or "openai" in api_base_url.lower():
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        action_json = response.choices[0].message.content
        action_json_clean = action_json.strip()
        if "```json" in action_json_clean:
            action_json_clean = action_json_clean.split("```json")[-1].split("```")[0]
        elif "```" in action_json_clean:
             action_json_clean = action_json_clean.split("```")[1]

        action_dict = json.loads(action_json_clean)
        if 'type' not in action_dict:
            action_dict['type'] = 'wait'
        return action_adapter.validate_python(action_dict)
    except Exception as e:
        print(f"[ERROR] Agent API or format error: {e}")
        return action_adapter.validate_python({'type': 'wait'})

def run_task(make_env, grade_fn, desc, seed, task_name):
    # Logs STRICTLY formatted with [START], [STEP], and [END] as mandated
    print(f"[START] {task_name} Task")
    env = make_env(seed=seed)
    obs = env.reset()
    done = False
    
    while not done:
        action = agent_act(obs)
        print(f"[STEP] Action selected: {action.type}")
        obs, reward, done, info = env.step(action)
        
    score = grade_fn(env)
    print(f"[END] Score/Reward: {score:.3f}\n")

def main():
    for name, make_env, grade_fn, desc in TASKS:
        run_task(make_env, grade_fn, desc, seed=42, task_name=name)

if __name__ == "__main__":
    main()
