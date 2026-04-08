import pytest
from triage_env.environment import ERSimulationEnv
from triage_env.models import HospitalResources, Patient, Vitals, AssignPriorityAction

def test_environment_initialization():
    initial_patients = [
        Patient(
            patient_id=1,
            symptoms=["chest pain"],
            vitals=Vitals(heart_rate=110, blood_pressure="140/90", oxygen_saturation=95.0, temperature=37.0),
            severity_indicators={"critical_vitals": True},
            arrival_time=0,
            deterioration_risk_score=0.8
        )
    ]
    initial_resources = HospitalResources(icu_beds=5, doctors_available=10)
    
    env = ERSimulationEnv(initial_patients, initial_resources)
    obs = env.reset()
    
    assert len(obs.patients) == 1
    assert obs.resources.icu_beds == 5
    assert obs.timestep == 0

def test_assign_priority_action():
    initial_patients = [
        Patient(
            patient_id=1,
            symptoms=["chest pain"],
            vitals=Vitals(heart_rate=110, blood_pressure="140/90", oxygen_saturation=95.0, temperature=37.0),
            severity_indicators={"critical_vitals": True},
            arrival_time=0,
            deterioration_risk_score=0.8
        )
    ]
    initial_resources = HospitalResources(icu_beds=5, doctors_available=10)
    
    env = ERSimulationEnv(initial_patients, initial_resources)
    env.reset()
    action = AssignPriorityAction(patient_id=1, esi_level=1)
    
    obs, reward, done, info = env.step(action)
    assert reward.value > 0  # Should be rewarded for correctly assigning level 1 to critical patient
    assert env.timestep == 1
