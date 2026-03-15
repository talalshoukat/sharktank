from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    transaction_trace_id: int = Field(
        ...,
        description="The TRANSACTIONTRACEID from T_TRANSACTIONTRACE",
        example=12345678,
    )


class BatchScoreRequest(BaseModel):
    transaction_trace_ids: list[int] = Field(
        ...,
        description="List of TRANSACTIONTRACEIDs to score",
        min_length=1,
        max_length=100,
    )
