"""Hatchet task for evaluating Twitter/X posts for publish readiness."""

from __future__ import annotations

from typing import Any

from hatchet_sdk import Context
from openai import OpenAI
from pydantic import BaseModel, Field

from hatchet_client import hatchet

DEFAULT_MODEL = "gpt-4o-mini"


class JudgeTweetInput(BaseModel):
    """Validated payload for the ``judge_tweet`` task."""

    tweet: str = Field(
        ..., description="Tweet text that needs to be evaluated before publishing."
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="OpenAI chat completion model to use for judgment.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0,
        le=2,
        description="Sampling temperature for the model when generating feedback.",
    )


class JudgeTweetResult(BaseModel):
    """Structured response from the ``judge_tweet`` task."""

    should_publish: bool
    feedback: str
    model: str


@hatchet.task(name="twitter.judge-tweet", input_validator=JudgeTweetInput)
def judge_tweet(input: JudgeTweetInput, ctx: Context) -> JudgeTweetResult:
    """Judge whether a tweet is ready to publish and provide feedback if not."""

    client = OpenAI()

    system_prompt = (
        "You are a meticulous social media editor specializing in Twitter/X. "
        "Assess the provided tweet for publish readiness, considering clarity, engagement, "
        "brand safety, length limits, and tone."
    )

    user_prompt = (
        "Review the following tweet and decide if it should be published as-is.\n\n"
        f"Tweet:\n{input.tweet}\n\n"
        "Respond with should_publish=true only when no changes are required. "
        "If changes are needed, set should_publish=false and give concise, actionable feedback "
        "focused on how to improve the tweet."
    )

    completion = client.responses.create(
        model=input.model,
        temperature=input.temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "JudgeTweetResponse",
                "schema": {
                    "type": "object",
                    "properties": {
                        "should_publish": {"type": "boolean"},
                        "feedback": {"type": "string", "maxLength": 500},
                    },
                    "required": ["should_publish", "feedback"],
                    "additionalProperties": False,
                },
            },
        },
        input=[
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    )

    parsed = _extract_response_json(completion)

    should_publish = bool(parsed.get("should_publish", False))
    feedback = parsed.get("feedback", "").strip()

    if should_publish:
        feedback = ""
    elif not feedback:
        feedback = "Revise the tweet to improve clarity, tone, or engagement before publishing."

    result = JudgeTweetResult(
        should_publish=should_publish,
        feedback=feedback,
        model=input.model,
    )

    ctx.log(f"Tweet judged as {should_publish} using model `{input.model}`.")

    return result.model_dump()


def _extract_response_json(completion: Any) -> dict[str, Any]:
    """Extract JSON content from the OpenAI responses API reply."""

    import json

    try:
        output_text = completion.output_text
    except AttributeError:
        try:
            output_text = completion.output[0].content[0].text  # type: ignore[index]
        except (AttributeError, IndexError) as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenAI returned an unexpected response structure."
            ) from exc

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive parsing
        raise RuntimeError("OpenAI response JSON could not be parsed.") from exc
