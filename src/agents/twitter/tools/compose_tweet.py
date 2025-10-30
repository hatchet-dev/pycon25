from hatchet_sdk import ConcurrencyExpression, ConcurrencyLimitStrategy, Context
from pydantic import BaseModel, Field

from common.dependencies import OpenAIDependency
from common.llm import generate
from hatchet_client import hatchet

ConcurrencyExpression(
    expression="'constant'",
    max_runs=1,
    limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
)

SYSTEM_PROMPT = (
    "You are an expert Twitter/X copywriter. You craft concise posts that stay within "
    "the 280-character limit, use strong hooks, and match the specified tone. Keep the "
    "language conversational, avoid excessive emojis, and ensure any line breaks are purposeful."
)


class ComposeTweetInput(BaseModel):
    prompt: str = Field(..., description="Core idea or instructions for the tweet.")

    previous_feedback: str | None = Field(
        default=None,
        description="Optional feedback from prior tweet evaluations to improve upon.",
    )

    previous_tweet: str | None = Field(
        default=None,
        description="Optional prior tweet text to revise based on feedback.",
    )


class ComposeTweetResponse(BaseModel):
    tweet: str
    hashtags: list[str]


@hatchet.task(name="twitter.compose-tweet", input_validator=ComposeTweetInput)
async def compose_tweet(
    input: ComposeTweetInput,
    _ctx: Context,
    openai: OpenAIDependency,
) -> ComposeTweetResponse:
    hashtag_instruction = "Include up to three relevant hashtags separated by spaces at the end of the tweet."

    user_prompt = (
        f"Compose a tweet/X post about: {input.prompt}\n"
        f"Requirements: {hashtag_instruction}\n"
        "Return the result as a JSON object with keys `tweet` and `hashtags`.\n"
        f"You've previously received the following feedback on the last iteration of the tweet: {input.previous_feedback}\n"
        f"The last iteration of the tweet was: {input.previous_tweet}"
    )

    return await generate(
        openai=openai,
        response_model=ComposeTweetResponse,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
