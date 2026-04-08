import random
from typing import List, Dict, Any
from .models import Patient, Vitals

SYMPTOM_PROFILES = {
    'chest pain': {
        'hr_range': (90, 140),
        'bp_range': [(90, 60), (160, 100)],
        'o2_range': (88, 99),
        'temp_range': (36.0, 37.5),
        'risk': 0.7
    },
    'fever': {
        'hr_range': (80, 120),
        'bp_range': [(100, 60), (140, 90)],
        'o2_range': (92, 100),
        'temp_range': (38.0, 40.0),
        'risk': 0.4
    },
    'trauma': {
        'hr_range': (100, 160),
        'bp_range': [(80, 50), (130, 90)],
        'o2_range': (85, 98),
        'temp_range': (35.0, 37.0),
        'risk': 0.8
    },
    'infection': {
        'hr_range': (90, 130),
        'bp_range': [(90, 60), (130, 90)],
        'o2_range': (90, 99),
        'temp_range': (38.0, 41.0),
        'risk': 0.6
    },
    'stable': {
        'hr_range': (60, 90),
        'bp_range': [(110, 70), (130, 85)],
        'o2_range': (97, 100),
        'temp_range': (36.5, 37.2),
        'risk': 0.1
    }
}

class PatientGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.patient_id_counter = 0

    def generate_patient(self, symptom: str, arrival_time: int) -> Patient:
        profile = SYMPTOM_PROFILES.get(symptom, SYMPTOM_PROFILES['stable'])
        hr = self.rng.randint(*profile['hr_range'])
        bp_sys = self.rng.randint(profile['bp_range'][0][0], profile['bp_range'][1][0])
        bp_dia = self.rng.randint(profile['bp_range'][0][1], profile['bp_range'][1][1])
        o2 = round(self.rng.uniform(*profile['o2_range']), 1)
        temp = round(self.rng.uniform(*profile['temp_range']), 1)
        risk = profile['risk'] + self.rng.uniform(-0.1, 0.1)
        patient = Patient(
            patient_id=self.patient_id_counter,
            symptoms=[symptom],
            vitals=Vitals(
                heart_rate=hr,
                blood_pressure=f"{bp_sys}/{bp_dia}",
                oxygen_saturation=o2,
                temperature=temp
            ),
            severity_indicators={
                'critical_vitals': hr > 130 or o2 < 90 or temp > 39.5 or bp_sys < 90,
                'trauma': symptom == 'trauma',
                'infection': symptom == 'infection',
                'chest_pain': symptom == 'chest pain',
                'fever': symptom == 'fever',
            },
            arrival_time=arrival_time,
            deterioration_risk_score=max(0.0, min(1.0, risk)),
            allocated_resources=[],
            deteriorated=False
        )
        self.patient_id_counter += 1
        return patient

    def generate_patients(self, n: int, symptoms: List[str], arrival_time: int) -> List[Patient]:
        return [self.generate_patient(self.rng.choice(symptoms), arrival_time) for _ in range(n)]
