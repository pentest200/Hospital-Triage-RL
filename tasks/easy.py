from triage_env.models import Vitals, Patient, HospitalResources
from triage_env.patient_generator import PatientGenerator
from triage_env.environment import ERSimulationEnv

def make_easy_env(seed=42):
    gen = PatientGenerator(seed)
    patients = [
        gen.generate_patient('chest pain', arrival_time=0),
        gen.generate_patient('fever', arrival_time=0),
        gen.generate_patient('stable', arrival_time=0)
    ]
    resources = HospitalResources(icu_beds=2, doctors_available=3)
    return ERSimulationEnv(patients, resources, max_timesteps=10, seed=seed)

description = """
EASY TASK:
- 3 patients
- Clear symptoms
- Minimal deterioration risk
- Ample resources
"""

def grade(env: ERSimulationEnv):
    # Score based on correct ESI assignment
    from triage_env.grader import grade_episode
    return grade_episode(env.patients, env.assigned_priorities)
