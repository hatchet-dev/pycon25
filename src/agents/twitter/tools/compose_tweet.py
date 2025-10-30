"""Hatchet task for crafting platform-appropriate Twitter/X posts."""

from __future__ import annotations

from hatchet_sdk import Context
from openai import OpenAI
from pydantic import BaseModel, Field

from common.response import response_to_pydantic
from hatchet_client import hatchet

DEFAULT_MODEL = "gpt-4o-mini"


class ComposeTweetInput(BaseModel):
    """Validated payload for the ``compose_tweet`` task."""

    prompt: str = Field(..., description="Core idea or instructions for the tweet.")
    tone: str = Field(
        default="punchy",
        description="Desired tone, e.g. punchy, witty, informative.",
    )
    include_hashtags: bool = Field(
        default=True,
        description="Whether to include up to 3 relevant hashtags.",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="OpenAI chat completion model to use for composition.",
    )
    temperature: float = Field(
        default=0.8,
        ge=0,
        le=2,
        description="Sampling temperature for the model.",
    )


class ComposeTweetResult(BaseModel):
    """Structured response from the ``compose_tweet`` task."""

    tweet: str
    tone: str
    hashtags: list[str]
    model: str


class ComposeTweetResponse(BaseModel):
    tweet: str
    hashtags: list[str]


@hatchet.task(name="twitter.compose-tweet", input_validator=ComposeTweetInput)
def compose_tweet(input: ComposeTweetInput, ctx: Context) -> ComposeTweetResult:
    """Generate a platform-tailored tweet/X post."""

    client = OpenAI()

    hashtag_instruction = (
        "Include up to three relevant hashtags separated by spaces at the end of the tweet."
        if input.include_hashtags
        else "Do not include hashtags."
    )

    system_prompt = (
        "You are an expert Twitter/X copywriter. You craft concise posts that stay within "
        "the 280-character limit, use strong hooks, and match the specified tone. Keep the "
        "language conversational, avoid excessive emojis, and ensure any line breaks are purposeful."
    )

    user_prompt = (
        f"Compose a tweet/X post about: {input.prompt}\n"
        f"Tone: {input.tone}\n"
        f"Requirements: {hashtag_instruction}\n"
        "Return the result as a JSON object with keys `tweet` and `hashtags`."
    )

    completion = client.chat.completions.create(
        model=input.model,
        temperature=input.temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "compose_tweet_response",
                "schema": ComposeTweetResponse.model_json_schema(),
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

    parsed = response_to_pydantic(completion, ComposeTweetResponse)

    tweet = parsed.tweet.strip()
    hashtags = [tag.strip() for tag in parsed.hashtags if tag.strip()]

    return ComposeTweetResult(
        tweet=tweet,
        tone=input.tone,
        hashtags=hashtags,
        model=input.model,
    )
