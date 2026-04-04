---
title: Dynamic ER Triage Simulation (ESI-Based)
emoji: 🚑
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - triage
  - simulation
---

# Dynamic ER Triage Simulation Environment (ESI-Based)

A production-grade OpenEnv environment for realistic emergency room triage using the Emergency Severity Index (ESI). Features dynamic patient deterioration, resource constraints, and multi-step decision-making. Designed for AI agents to learn and compete in realistic ER triage scenarios.

## Features
- Dynamic, stochastic patient deterioration
- Multi-step, non-trivial action space
- Dense, explainable reward system
- Three tasks (easy, medium, hard)
- OpenEnv-compliant (step, reset, state, render)
- Docker-ready, reproducible

## Usage
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python scripts/run_baseline.py
```

## Citation
If you use this environment, please cite the OpenEnv Hackathon Team.
