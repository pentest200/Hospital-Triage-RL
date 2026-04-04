# Dynamic ER Triage Simulation Environment (ESI-Based)

## Overview
A production-grade OpenEnv environment simulating a real-world emergency room triage using the Emergency Severity Index (ESI). The environment features dynamic patient deterioration, resource constraints, and multi-step decision-making. Designed for AI agents to learn and compete in realistic ER triage scenarios.

## Why is this useful?
- **Real-world relevance:** Models the complexity of ER triage, where prioritization and resource allocation can save lives.
- **Dynamic & interactive:** Patient conditions change over time, requiring adaptive, multi-step strategies.
- **Benchmarking:** Provides a rigorous, reproducible testbed for AI/ML research in healthcare operations.

## Action & Observation Space
### Observation
- **patients:** List of patients, each with:
  - `patient_id`, `symptoms`, `vitals` (heart_rate, blood_pressure, oxygen_saturation, temperature), `severity_indicators`, `arrival_time`, `current_priority`, `deterioration_risk_score`
- **resources:** ICU beds, doctors available
- **timestep:** Current simulation step

### Action
- `assign_priority(patient_id, esi_level)`
- `reassign_priority(patient_id, new_level)`
- `allocate_resource(patient_id, resource_type)`
- `escalate_patient(patient_id)`
- `wait` (advance time)

## Reward Design
- **Correct prioritization:** +0.5
- **Early detection of critical patient:** +0.5 bonus
- **Late action on deteriorating patient:** Penalty
- **Wrong prioritization:** Negative reward
- **Efficient resource allocation:** Positive reward
- **Wasting resources:** Penalty
- **Dense, meaningful rewards** (not sparse)

## Tasks
### Easy
- Few patients
- Clear symptoms
- Minimal deterioration
- Ample resources

### Medium
- Mixed patient severity
- Some deterioration
- Resource constraints begin

### Hard
- Many patients
- Rapid deterioration
- Severe resource scarcity
- Requires long-term planning

## Setup Instructions
1. **Clone repo & install dependencies:**
   ```sh
   pip install -r requirements.txt  # or see openenv.yaml dependencies
   ```
2. **Set OpenAI API key:**
   ```sh
   export OPENAI_API_KEY=sk-...
   ```
3. **Run baseline:**
   ```sh
   python scripts/run_baseline.py
   ```
4. **Docker:**
   ```sh
   docker build .
   docker run .
   ```

## Baseline Scores
The baseline script uses GPT-4 to interact with the environment. Example output:
```
--- Easy Task ---
Steps: 5, Total Reward: 1.20, Grader Score: 1.000

--- Medium Task ---
Steps: 7, Total Reward: 0.80, Grader Score: 0.875

--- Hard Task ---
Steps: 12, Total Reward: 0.30, Grader Score: 0.714
```

## OpenEnv Compliance
- Implements `step`, `reset`, `state` methods
- Typed Pydantic models for Observation, Action, Reward
- Includes `openenv.yaml` with metadata
- Passes `openenv validate`

## Authors
- Ujjwal Shreshtha

## License
- MIT
