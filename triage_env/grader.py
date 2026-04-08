from typing import List, Dict, Any
from .models import Patient

# ESI Decision Tree Logic
# ESI 1: Immediate life-saving intervention required
# ESI 2: High risk, confused/lethargic/disoriented, severe pain/distress
# ESI 3: Multiple resources needed
# ESI 4: One resource
# ESI 5: No resources

def compute_esi_level(patient: Patient) -> int:
    v = patient.vitals
    # ESI 1: Critical vitals or trauma
    if patient.severity_indicators.get('critical_vitals', False) or v.oxygen_saturation < 90 or v.heart_rate > 140:
        return 1
    # ESI 2: High risk (e.g., chest pain, infection with fever, trauma)
    if patient.severity_indicators.get('chest_pain', False) or patient.severity_indicators.get('trauma', False):
        return 2
    if patient.severity_indicators.get('infection', False) and v.temperature > 38.5:
        return 2
    # ESI 3: Needs multiple resources (simulate by high risk score)
    if patient.deterioration_risk_score > 0.6:
        return 3
    # ESI 4: Needs one resource
    if patient.deterioration_risk_score > 0.3:
        return 4
    # ESI 5: Stable
    return 5

def grade_action(patient: Patient, action_level: int) -> float:
    correct_level = compute_esi_level(patient)
    if action_level == correct_level:
        return 1.0
    elif abs(action_level - correct_level) == 1:
        return 0.5
    elif correct_level == 1 and action_level > 2:
        return -1.0  # Strong penalty for missing critical
    else:
        return 0.0

def grade_episode(patients: List[Patient], actions: Dict[int, int]) -> float:
    # actions: patient_id -> assigned_level
    total = 0.0
    for p in patients:
        assigned = actions.get(p.patient_id, None)
        if assigned is not None:
            total += grade_action(p, assigned)
    raw_score = max(0.0, min(1.0, total / len(patients) if patients else 0.0))
    # Scale to (0.01, 0.99) to satisfy "strictly between 0 and 1" requirement
    return 0.01 + (raw_score * 0.98)
