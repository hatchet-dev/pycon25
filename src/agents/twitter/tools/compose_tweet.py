"""Hatchet task for crafting platform-appropriate Twitter/X posts."""

from __future__ import annotations

from typing import Any

from hatchet_sdk import Context
from openai import OpenAI
from pydantic import BaseModel, Field

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

    completion = client.responses.create(
        model=input.model,
        temperature=input.temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ComposeTweetResponse",
                "schema": {
                    "type": "object",
                    "properties": {
                        "tweet": {"type": "string", "maxLength": 280},
                        "hashtags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tweet", "hashtags"],
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

    tweet = parsed["tweet"].strip()
    hashtags = [tag.strip() for tag in parsed.get("hashtags", []) if tag.strip()]

    result = ComposeTweetResult(
        tweet=tweet,
        tone=input.tone,
        hashtags=hashtags,
        model=input.model,
    )

    ctx.log(
        f"Generated tweet with {len(tweet)} characters using model `{input.model}`."
    )

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
