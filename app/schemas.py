from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    Password: list[str]

    @field_validator("Password")
    @classmethod
    def validate_passwords(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Password list must not be empty")

        for password in value:
            if not isinstance(password, str) or not password.strip():
                raise ValueError("Each password must be a non-empty string")

        return value


class PredictResponse(BaseModel):
    Times: list[float]


class TriggerRequest(BaseModel):
    data_url: str

    @field_validator("data_url")
    @classmethod
    def validate_data_url(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("data_url must not be empty")

        if not value.startswith(("http://", "https://")):
            raise ValueError("data_url must start with http:// or https://")

        return value


class TriggerResponse(BaseModel):
    status: str
    pipeline_id: int | None = None
    message: str | None = None


class ReloadResponse(BaseModel):
    status: str
    model_name: str | None = None
    model_alias: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
