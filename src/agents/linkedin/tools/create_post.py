from hatchet_sdk import Context
from pydantic import BaseModel, Field

from common.dependencies import OpenAIDependency
from common.response import response_to_pydantic
from hatchet_client import hatchet

DEFAULT_MODEL = "gpt-4o-mini"


class CreatePostInput(BaseModel):
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
    headline: str
    body: str
    cta: str
    hashtags: list[str]
    post: str
    tone: str
    audience: str
    model: str
    temperature: float


class CreateLinkedInPostResponse(BaseModel):
    headline: str
    body: str
    cta: str
    hashtags: list[str]


SYSTEM_PROMPT = (
    "You are an executive LinkedIn content strategist. Write thoughtful posts that open "
    "with a bold headline, deliver actionable insight in two short paragraphs, and close "
    "with a motivating call to action addressing the specified audience. Maintain a "
    "professional yet personable voice and weave in concrete details where possible."
)


@hatchet.task(name="linkedin.create-post", input_validator=CreatePostInput)
async def create_linkedin_post(
    input: CreatePostInput, _ctx: Context, openai: OpenAIDependency
) -> CreatePostResult:
    hashtag_guidance = (
        "Include a final line with 1-3 relevant hashtags tailored to LinkedIn readers."
        if input.include_hashtags
        else "Do not include hashtags in the final post."
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

    completion = await openai.chat.completions.create(
        model=input.model,
        temperature=float(input.temperature),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "create_linkedin_post_response",
                "schema": CreateLinkedInPostResponse.model_json_schema(),
            },
        },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    parsed_payload = response_to_pydantic(completion, CreateLinkedInPostResponse)

    if not input.include_hashtags:
        parsed_payload.hashtags = []

    composed_post = _compose_linkedin_post(parsed_payload)

    return CreatePostResult(
        headline=parsed_payload.headline.strip(),
        body=parsed_payload.body.strip(),
        cta=parsed_payload.cta.strip(),
        hashtags=[tag.strip() for tag in parsed_payload.hashtags if tag.strip()],
        post=composed_post,
        tone=input.tone,
        audience=input.audience,
        model=input.model,
        temperature=float(input.temperature),
    )


def _compose_linkedin_post(post: CreateLinkedInPostResponse) -> str:
    sections = [post.headline.strip(), "", post.body.strip(), "", post.cta.strip()]

    if post.hashtags:
        sections.append("")
        sections.append(" ".join(post.hashtags))

    return "\n".join(section for section in sections if section)
