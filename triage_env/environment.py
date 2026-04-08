from typing import Tuple, Dict, Any
from .models import Observation, Action, Reward, Patient, HospitalResources
from .patient_generator import PatientGenerator
from .grader import compute_esi_level, grade_action
import copy
import random


class ERSimulationEnv:
    """
    OpenEnv-compliant ER Triage Simulation Environment (ESI-Based)
    Implements step, reset, state, and render methods. Modular reward system. Fully typed.
    """
    def __init__(self, initial_patients: list, initial_resources: HospitalResources, max_timesteps: int = 30, seed: int = 42):
        self.generator = PatientGenerator(seed)
        self.initial_patients = copy.deepcopy(initial_patients)
        self.initial_resources = copy.deepcopy(initial_resources)
        self.max_timesteps = max_timesteps
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> Observation:
        """Reset environment to initial state."""
        self.patients = copy.deepcopy(self.initial_patients)
        self.resources = copy.deepcopy(self.initial_resources)
        self.timestep = 0
        self.done = False
        self.patient_map = {p.patient_id: p for p in self.patients}
        self.action_history = []
        self.assigned_priorities = {}  # patient_id -> esi_level
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Take an action, advance the environment, and return (observation, reward, done, info)."""
        assert not self.done, "Episode is done. Call reset()."
        reward, reward_details = self._calculate_reward(action)
        self.action_history.append(action)
        self.timestep += 1
        self._simulate_patient_deterioration()
        # Done if max timesteps or all patients assigned and stabilized
        if self.timestep >= self.max_timesteps or all(self.assigned_priorities.get(p.patient_id) is not None for p in self.patients):
            self.done = True
        obs = self._get_observation()
        # Clamp reward strictly within (0, 1) for OpenEnv compliance
        clamped_reward = max(0.01, min(0.99, (reward + 2.0) / 4.0))
        return obs, Reward(value=clamped_reward, details=reward_details), self.done, {}

    def _calculate_reward(self, action: Action) -> Tuple[float, dict]:
        """Calculate reward for the given action and update patient state."""
        reward = 0.0
        details = {}
        # Assign priority
        if action.type == 'assign_priority':
            pid = action.patient_id
            level = action.esi_level
            if pid in self.patient_map:
                self.assigned_priorities[pid] = level
                patient = self.patient_map[pid]
                patient.current_priority = level
                correct_level = compute_esi_level(patient)
                if level == correct_level:
                    reward += 0.5
                    details['assign_priority'] = 0.5
                else:
                    reward -= 0.5
                    details['assign_priority'] = -0.5
                if correct_level == 1 and self.timestep <= 2 and level == 1:
                    reward += 0.5
                    details['early_critical'] = 0.5
        elif action.type == 'reassign_priority':
            pid = action.patient_id
            level = action.new_level
            if pid in self.patient_map:
                self.assigned_priorities[pid] = level
                patient = self.patient_map[pid]
                patient.current_priority = level
                correct_level = compute_esi_level(patient)
                if level == correct_level:
                    reward += 0.3
                    details['reassign_priority'] = 0.3
                else:
                    reward -= 0.3
                    details['reassign_priority'] = -0.3
        elif action.type == 'allocate_resource':
            pid = action.patient_id
            resource = action.resource_type
            if pid in self.patient_map:
                patient = self.patient_map[pid]
                if resource == 'icu_bed' and self.resources.icu_beds > 0:
                    self.resources.icu_beds -= 1
                    patient.allocated_resources.append('icu_bed')
                    reward += 0.3
                    details['allocate_icu'] = 0.3
                elif resource == 'doctor' and self.resources.doctors_available > 0:
                    self.resources.doctors_available -= 1
                    patient.allocated_resources.append('doctor')
                    reward += 0.2
                    details['allocate_doctor'] = 0.2
                else:
                    reward -= 0.2
                    details['waste_resource'] = -0.2
        elif action.type == 'escalate_patient':
            pid = action.patient_id
            if pid in self.patient_map:
                patient = self.patient_map[pid]
                if patient.current_priority and patient.current_priority > 1:
                    patient.current_priority = 1
                    self.assigned_priorities[pid] = 1
                    reward += 0.5
                    details['escalate'] = 0.5
                else:
                    reward -= 0.2
                    details['escalate'] = -0.2
        elif action.type == 'wait':
            details['wait'] = 0.0
        else:
            reward -= 0.1
            details['unknown_action'] = -0.1
        # Penalty for late action on deteriorating patient
        for p in self.patients:
            if p.deteriorated and (p.patient_id not in self.assigned_priorities or self.assigned_priorities[p.patient_id] > 2):
                reward -= 0.5
                details[f'late_{p.patient_id}'] = -0.5
        # Penalty for wasting ICU on non-critical
        for p in self.patients:
            if 'icu_bed' in p.allocated_resources and compute_esi_level(p) > 2:
                reward -= 0.3
                details[f'waste_icu_{p.patient_id}'] = -0.3
        return reward, details

    def _simulate_patient_deterioration(self):
        """Stochastically deteriorate patient vitals based on risk."""
        for p in self.patients:
            if not p.deteriorated:
                risk = p.deterioration_risk_score
                if self.rng.random() < risk * 0.2:
                    p.vitals.heart_rate = min(180, p.vitals.heart_rate + self.rng.randint(5, 20))
                    sys, dia = map(int, p.vitals.blood_pressure.split('/'))
                    sys = max(60, sys - self.rng.randint(0, 10))
                    dia = max(40, dia - self.rng.randint(0, 5))
                    p.vitals.blood_pressure = f"{sys}/{dia}"
                    p.vitals.oxygen_saturation = max(70.0, p.vitals.oxygen_saturation - self.rng.uniform(0, 2))
                    p.vitals.temperature = min(42.0, p.vitals.temperature + self.rng.uniform(0, 0.5))
                    p.deterioration_risk_score = min(1.0, p.deterioration_risk_score + self.rng.uniform(0.05, 0.2))
                    p.deteriorated = True
                    p.severity_indicators['critical_vitals'] = True


    def _get_observation(self) -> Observation:
        """Return a deep copy of the current observation."""
        return Observation(
            patients=copy.deepcopy(self.patients),
            resources=copy.deepcopy(self.resources),
            timestep=self.timestep
        )

    def state(self) -> Dict[str, Any]:
        """Return the internal state for reproducibility/debugging."""
        return {
            'patients': [p.model_dump() for p in self.patients],
            'resources': self.resources.model_dump(),
            'timestep': self.timestep,
            'assigned_priorities': self.assigned_priorities,
            'done': self.done
        }

    def render(self) -> None:
        """Print a human-readable summary of the environment state."""
        print(f"\n--- Timestep {self.timestep} ---")
        for p in self.patients:
            print(f"Patient {p.patient_id}: symptoms={p.symptoms}, vitals={p.vitals.model_dump()}, priority={p.current_priority}, risk={p.deterioration_risk_score:.2f}, deteriorated={p.deteriorated}, resources={p.allocated_resources}")
        print(f"Resources: ICU beds={self.resources.icu_beds}, Doctors={self.resources.doctors_available}")
        print(f"Done: {self.done}")
