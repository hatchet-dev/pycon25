from datetime import datetime, timezone

from hatchet_sdk import Context
from pydantic import BaseModel, Field

from hatchet_client import hatchet


class SimulatePostInput(BaseModel):
    post: str = Field(..., description="Post content to simulate publishing.")
    channel: str = Field(
        default="linkedin",
        description="Target channel for the simulated post.",
    )
    schedule: datetime | None = Field(
        default=None,
        description="Scheduled time in ISO8601 format or datetime for the post.",
    )


class SimulationResult(BaseModel):
    post: str
    scheduled_time: datetime = Field(
        description="The scheduled time in UTC when the post would go live."
    )
    channel: str
    status: str

    model_config = {
        "json_encoders": {datetime: lambda value: value.isoformat()},
    }


@hatchet.task(name="linkedin.simulate-post", input_validator=SimulatePostInput)
def simulate_linkedin_post(input: SimulatePostInput, _ctx: Context) -> SimulationResult:
    return SimulationResult(
        post=input.post,
        scheduled_time=input.schedule or datetime.now(timezone.utc),
        channel=input.channel,
        status="simulated",
    )
