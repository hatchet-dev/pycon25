from hatchet_sdk import Context
from pydantic import BaseModel, Field

from common.dependencies import OpenAIDependency
from common.llm import generate
from hatchet_client import hatchet

SYSTEM_PROMPT = (
    "You are a meticulous social media editor specializing in Twitter/X. "
    "Assess the provided tweet for publish readiness, considering clarity, engagement, "
    "brand safety, length limits, and tone."
)


class JudgeTweetInput(BaseModel):
    tweet: str = Field(
        ..., description="Tweet text that needs to be evaluated before publishing."
    )


class JudgeTweetResponse(BaseModel):
    should_publish: bool
    feedback: str


@hatchet.task(name="twitter.judge-tweet", input_validator=JudgeTweetInput)
async def judge_tweet(
    input: JudgeTweetInput, _ctx: Context, openai: OpenAIDependency
) -> JudgeTweetResponse:
    user_prompt = (
        "Review the following tweet and decide if it should be published as-is.\n\n"
        f"Tweet:\n{input.tweet}\n\n"
        "Respond with should_publish=true only when no changes are required. "
        "If changes are needed, set should_publish=false and give concise, actionable feedback "
        "focused on how to improve the tweet."
    )

    return await generate(
        openai=openai,
        response_model=JudgeTweetResponse,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
