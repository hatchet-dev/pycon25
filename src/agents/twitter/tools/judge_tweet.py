"""Hatchet task for evaluating Twitter/X posts for publish readiness."""

from __future__ import annotations

from hatchet_sdk import Context
from openai import OpenAI
from pydantic import BaseModel, Field

from common.response import response_to_pydantic
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


class JudgeTweetResponse(BaseModel):
    should_publish: bool
    feedback: str


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

    completion = client.chat.completions.create(
        model=input.model,
        temperature=input.temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_tweet_response",
                "schema": JudgeTweetResponse.model_json_schema(),
            },
        },
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    parsed = response_to_pydantic(completion, JudgeTweetResponse)

    should_publish = parsed.should_publish
    feedback = parsed.feedback

    if should_publish:
        feedback = ""
    elif not feedback:
        feedback = "Revise the tweet to improve clarity, tone, or engagement before publishing."

    return JudgeTweetResult(
        should_publish=should_publish,
        feedback=feedback,
        model=input.model,
    )
