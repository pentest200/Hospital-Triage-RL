from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class Vitals(BaseModel):
    heart_rate: int
    blood_pressure: str  # e.g., "120/80"
    oxygen_saturation: float
    temperature: float

class Patient(BaseModel):
    patient_id: int
    symptoms: List[str]
    vitals: Vitals
    severity_indicators: Dict[str, bool]
    arrival_time: int
    current_priority: Optional[int] = None  # ESI level 1-5
    deterioration_risk_score: float
    allocated_resources: List[str] = Field(default_factory=list)
    deteriorated: bool = False

class HospitalResources(BaseModel):
    icu_beds: int
    doctors_available: int

class Observation(BaseModel):
    patients: List[Patient]
    resources: HospitalResources
    timestep: int

class AssignPriorityAction(BaseModel):
    type: Literal['assign_priority'] = 'assign_priority'
    patient_id: int
    esi_level: int

class ReassignPriorityAction(BaseModel):
    type: Literal['reassign_priority'] = 'reassign_priority'
    patient_id: int
    new_level: int

class AllocateResourceAction(BaseModel):
    type: Literal['allocate_resource'] = 'allocate_resource'
    patient_id: int
    resource_type: str

class EscalatePatientAction(BaseModel):
    type: Literal['escalate_patient'] = 'escalate_patient'
    patient_id: int

class WaitAction(BaseModel):
    type: Literal['wait'] = 'wait'

Action = AssignPriorityAction | ReassignPriorityAction | AllocateResourceAction | EscalatePatientAction | WaitAction

class Reward(BaseModel):
    value: float
    details: Dict[str, Any] = Field(default_factory=dict)
