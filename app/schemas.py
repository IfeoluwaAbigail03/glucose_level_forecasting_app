"""
Input stays the same 14 features — both models consume the identical
feature set. The response now carries both v1's predicted number and
v2's advisory alert.
"""
from pydantic import BaseModel, Field


class GlucosePredictionRequest(BaseModel):
    bg_past_0_00: float = Field(..., description="Current glucose (mmol/L)")
    bg_prev_1: float = Field(..., description="Glucose 5 min ago")
    bg_prev_2: float = Field(..., description="Glucose 10 min ago")
    bg_slope_15min: float
    bg_slope_60min: float
    bg_change_15min: float
    bg_change_60min: float
    bg_rolling_mean_3: float
    bg_rolling_std_3: float
    roc_15min: float
    roc_30min: float
    acceleration: float
    hour: int = Field(..., ge=0, le=23)
    hypo_risk: int = Field(..., ge=0, le=1)


class GlucosePredictionResponse(BaseModel):
    predicted_bg_1h: float
    clinical_flag: str  # "normal" | "hyper_risk" — derived from the v1 number
    hypo_alert: bool     # from v2's classifier — advisory only
    hypo_alert_probability: float
    advisory_note: str