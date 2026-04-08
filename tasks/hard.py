from triage_env.models import Vitals, Patient, HospitalResources
from triage_env.patient_generator import PatientGenerator
from triage_env.environment import ERSimulationEnv

def make_hard_env(seed=44):
    gen = PatientGenerator(seed)
    patients = [
        gen.generate_patient('trauma', arrival_time=0),
        gen.generate_patient('chest pain', arrival_time=0),
        gen.generate_patient('infection', arrival_time=0),
        gen.generate_patient('fever', arrival_time=0),
        gen.generate_patient('trauma', arrival_time=1),
        gen.generate_patient('infection', arrival_time=1),
        gen.generate_patient('stable', arrival_time=1)
    ]
    resources = HospitalResources(icu_beds=1, doctors_available=1)
    return ERSimulationEnv(patients, resources, max_timesteps=20, seed=seed)

description = """
HARD TASK:
- 7 patients
- Rapid deterioration
- Severe resource scarcity
- Requires long-term planning
"""

def grade(env: ERSimulationEnv):
    from triage_env.grader import grade_episode
    score = grade_episode(env.patients, env.assigned_priorities)
    return max(0.01, min(0.99, score))
