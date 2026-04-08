from triage_env.models import Vitals, Patient, HospitalResources
from triage_env.patient_generator import PatientGenerator
from triage_env.environment import ERSimulationEnv

def make_medium_env(seed=43):
    gen = PatientGenerator(seed)
    patients = [
        gen.generate_patient('trauma', arrival_time=0),
        gen.generate_patient('infection', arrival_time=0),
        gen.generate_patient('fever', arrival_time=0),
        gen.generate_patient('stable', arrival_time=0)
    ]
    resources = HospitalResources(icu_beds=1, doctors_available=2)
    return ERSimulationEnv(patients, resources, max_timesteps=15, seed=seed)

description = """
MEDIUM TASK:
- 4 patients
- Mixed severity
- Some deterioration risk
- Resource constraints begin
"""

def grade(env: ERSimulationEnv):
    from triage_env.grader import grade_episode
    score = grade_episode(env.patients, env.assigned_priorities)
    return max(0.01, min(0.99, score))
