"""Request/response models for the serving API.

The feature model mirrors the raw training columns (minus the target) so callers
send human-readable values and the serving pipeline handles encoding.
"""

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    checking_balance: str = Field(..., examples=["< 0 DM"])
    months_loan_duration: int = Field(..., examples=[12])
    credit_history: str = Field(..., examples=["good"])
    purpose: str = Field(..., examples=["furniture/appliances"])
    amount: int = Field(..., examples=[2096])
    savings_balance: str = Field(..., examples=["< 100 DM"])
    employment_duration: str = Field(..., examples=["1 - 4 years"])
    percent_of_income: int = Field(..., examples=[2])
    years_at_residence: int = Field(..., examples=[3])
    age: int = Field(..., examples=[35])
    other_credit: str = Field(..., examples=["none"])
    housing: str = Field(..., examples=["own"])
    existing_loans_count: int = Field(..., examples=[1])
    job: str = Field(..., examples=["skilled"])
    dependents: int = Field(..., examples=[1])
    phone: str = Field(..., examples=["yes"])


class PredictionResponse(BaseModel):
    default_probability: float
    prediction: str
    is_defaulter: bool
