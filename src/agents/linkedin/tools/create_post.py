"""Hatchet task for generating a LinkedIn-ready post with the OpenAI SDK."""

from __future__ import annotations

from typing import Any

from hatchet_sdk import Context

from hatchet_client import hatchet

from openai import OpenAI

from pydantic import BaseModel, Field


DEFAULT_MODEL = "gpt-4o-mini"


class CreatePostInput(BaseModel):
    """Validated payload for the ``create_linkedin_post`` task."""

    prompt: str = Field(..., description="Core idea or instructions for the post.")
    tone: str = Field(
        default="professional",
        description="Desired tone, e.g. professional, casual, conversational.",
    )
    audience: str = Field(
        default="LinkedIn audience",
        description="Target audience description.",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="OpenAI chat completion model to use.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2,
        description="Sampling temperature for the model.",
    )
    include_hashtags: bool = Field(
        default=True,
        description="Whether to add 1-3 relevant hashtags at the end of the post.",
    )


class CreatePostResult(BaseModel):
    """Structured response returned by ``create_linkedin_post``."""

    headline: str
    body: str
    cta: str
    hashtags: list[str]
    post: str
    tone: str
    audience: str
    model: str
    temperature: float


@hatchet.task(name="linkedin.create-post")
def create_linkedin_post(input: CreatePostInput, ctx: Context) -> dict[str, Any]:
    """Generate copy for a LinkedIn post.

    Expected ``input`` keys:
        - ``prompt``: core idea or instructions for the post (required)
        - ``tone``: desired tone, e.g. "professional", "casual" (optional)
        - ``audience``: target audience description (optional)
        - ``model``: OpenAI chat completion model override (optional)
        - ``temperature``: sampling temperature override (optional)
    """

    client = OpenAI()

    hashtag_guidance = (
        "Include a final line with 1-3 relevant hashtags tailored to LinkedIn readers."
        if input.include_hashtags
        else "Do not include hashtags in the final post."
    )

    system_instruction = (
        "You are an executive LinkedIn content strategist. Write thoughtful posts that open "
        "with a bold headline, deliver actionable insight in two short paragraphs, and close "
        "with a motivating call to action addressing the specified audience. Maintain a "
        "professional yet personable voice and weave in concrete details where possible."
    )

    user_prompt = (
        f"Topic prompt: {input.prompt}\n"
        f"Tone: {input.tone}\n"
        f"Audience: {input.audience}\n"
        "Guidelines: Keep the headline under 70 characters, limit the body to 2-3 short "
        "paragraphs (<=180 words total), make the CTA audience-specific, and ensure the "
        "overall post feels like a LinkedIn thought-leadership update. "
        f"{hashtag_guidance}\n"
        "Return a JSON object with keys `headline`, `body`, `cta`, and `hashtags` (array)."
    )

    completion = client.responses.create(
        model=input.model,
        temperature=float(input.temperature),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "CreateLinkedInPostResponse",
                "schema": {
                    "type": "object",
                    "properties": {
                        "headline": {"type": "string", "maxLength": 120},
                        "body": {"type": "string"},
                        "cta": {"type": "string"},
                        "hashtags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["headline", "body", "cta", "hashtags"],
                    "additionalProperties": False,
                },
            },
        },
        input=[
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    )

    parsed_payload = _extract_response_json(completion)

    if not input.include_hashtags:
        parsed_payload["hashtags"] = []

    composed_post = _compose_linkedin_post(
        parsed_payload["headline"],
        parsed_payload["body"],
        parsed_payload["cta"],
        parsed_payload.get("hashtags", []),
    )

    result = CreatePostResult(
        headline=parsed_payload["headline"].strip(),
        body=parsed_payload["body"].strip(),
        cta=parsed_payload["cta"].strip(),
        hashtags=[tag.strip() for tag in parsed_payload.get("hashtags", []) if tag.strip()],
        post=composed_post,
        tone=input.tone,
        audience=input.audience,
        model=input.model,
        temperature=float(input.temperature),
    )

    ctx.log(
        "Generated LinkedIn post with headline `%s` using model `%s`."
        % (result.headline, input.model)
    )

    return result.model_dump()


def _compose_linkedin_post(
    headline: str, body: str, cta: str, hashtags: list[str]
) -> str:
    """Assemble the final LinkedIn post string from structured components."""

    sections = [headline.strip(), "", body.strip(), "", cta.strip()]

    if hashtags:
        sections.append("")
        sections.append(" ".join(hashtags))

    return "\n".join(section for section in sections if section)


def _extract_response_json(completion: Any) -> dict[str, Any]:
    """Extract JSON content from the OpenAI responses API reply."""

    import json

    try:
        output_text = completion.output_text
    except AttributeError:
        try:
            output_text = completion.output[0].content[0].text  # type: ignore[index]
        except (AttributeError, IndexError) as exc:  # pragma: no cover
            raise RuntimeError("OpenAI returned an unexpected response structure.") from exc

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive parsing
        raise RuntimeError("OpenAI response JSON could not be parsed.") from exc


