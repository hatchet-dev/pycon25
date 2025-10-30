from hatchet_sdk import Context
from pydantic import BaseModel, Field

from common.dependencies import OpenAIDependency
from common.llm import generate
from hatchet_client import hatchet


class CreatePostInput(BaseModel):
    prompt: str = Field(..., description="Core idea or instructions for the post.")


class CreatePostResult(BaseModel):
    headline: str
    body: str
    cta: str
    hashtags: list[str]
    post: str


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
    user_prompt = (
        f"Topic prompt: {input.prompt}\n"
        "Guidelines: Keep the headline under 70 characters, limit the body to 2-3 short "
        "paragraphs (<=180 words total), make the CTA audience-specific, and ensure the "
        "overall post feels like a LinkedIn thought-leadership update. "
        "Include a final line with 1-3 relevant hashtags tailored to LinkedIn readers.\n"
        "Return a JSON object with keys `headline`, `body`, `cta`, and `hashtags` (array)."
    )

    completion = await generate(
        openai=openai,
        response_model=CreateLinkedInPostResponse,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    composed_post = _compose_linkedin_post(completion)

    return CreatePostResult(
        headline=completion.headline.strip(),
        body=completion.body.strip(),
        cta=completion.cta.strip(),
        hashtags=[tag.strip() for tag in completion.hashtags if tag.strip()],
        post=composed_post,
    )


def _compose_linkedin_post(post: CreateLinkedInPostResponse) -> str:
    sections = [post.headline.strip(), "", post.body.strip(), "", post.cta.strip()]

    if post.hashtags:
        sections.append("")
        sections.append(" ".join(post.hashtags))

    return "\n".join(section for section in sections if section)
