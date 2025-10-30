from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hatchet_sdk import Context
from pydantic import BaseModel, Field

from hatchet_client import hatchet


class SimulatePostInput(BaseModel):
    """Validated payload for the ``simulate_post`` task."""

    post: str = Field(..., description="Post content to simulate publishing.")
    channel: str = Field(
        default="linkedin",
        description="Target channel for the simulated post.",
    )
    schedule: datetime | str | None = Field(
        default=None,
        description="Scheduled time in ISO8601 format or datetime for the post.",
    )


class SimulationResult(BaseModel):
    """Structured response returned by the ``simulate_post`` task."""

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
def simulate_linkedin_post(input: SimulatePostInput, ctx: Context) -> dict[str, Any]:
    """Simulate sending a LinkedIn post to downstream systems.

    Expected ``input`` keys:
        - ``post``: post content to simulate (required)
        - ``channel``: target channel, defaults to "linkedin" (optional)
        - ``schedule``: ISO8601 string or ``datetime`` for scheduled posting time (optional)
    """

    post_content = input.post
    channel = input.channel
    schedule_input = input.schedule

    if isinstance(schedule_input, str):
        try:
            schedule_value = datetime.fromisoformat(schedule_input)
        except (
            ValueError
        ) as exc:  # pragma: no cover - bubble up invalid formats clearly
            raise ValueError(
                "`schedule` must be a valid ISO8601 datetime string."
            ) from exc
    else:
        schedule_value = schedule_input

    if schedule_value is None:
        scheduled_time = datetime.now(timezone.utc)
    else:
        if schedule_value.tzinfo is None:
            scheduled_time = schedule_value.replace(tzinfo=timezone.utc)
        else:
            scheduled_time = schedule_value.astimezone(timezone.utc)

    result = SimulationResult(
        post=post_content,
        scheduled_time=scheduled_time,
        channel=channel,
        status="simulated",
    )

    ctx.log(f"Simulated posting for channel `{channel}` at `{result.scheduled_time}`.")

    return result.model_dump(mode="json")
