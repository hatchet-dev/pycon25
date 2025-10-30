import re

import aiohttp
from hatchet_sdk import Context
from pydantic import BaseModel, Field, HttpUrl

from common.dependencies import OpenAIDependency
from common.llm import generate
from hatchet_client import hatchet

DEFAULT_MAX_CHARACTERS = 20_000


class ReadWebsiteInput(BaseModel):
    url: HttpUrl = Field(..., description="URL of the web page to retrieve.")
    timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description="Timeout for the HTTP request in seconds.",
    )

    max_characters: int = Field(
        default=DEFAULT_MAX_CHARACTERS,
        ge=1_000,
        le=100_000,
        description="Maximum number of HTML characters forwarded to the model.",
    )


class ReadWebsiteResultFromLLM(BaseModel):
    url: HttpUrl
    title: str
    content_markdown: str = Field(
        description="Primary page content converted to Markdown."
    )
    summary: str | None = Field(
        default=None, description="Optional short summary (≤3 sentences) of the page."
    )


class ReadWebsiteResult(ReadWebsiteResultFromLLM):
    url: HttpUrl


@hatchet.task(name="researcher.read-website", input_validator=ReadWebsiteInput)
async def read_website(
    input: ReadWebsiteInput, ctx: Context, openai: OpenAIDependency
) -> ReadWebsiteResult:
    ctx.log(
        f"Fetching URL `{input.url}` with timeout `{input.timeout_seconds}` seconds "
    )
    async with (
        aiohttp.ClientSession() as session,
        session.get(url=input.url.unicode_string()) as response,
    ):
        if response.status != 200:
            raise RuntimeError(
                f"Request to `{input.url}` failed with status code {response.status}."
            )

        text = await response.text()

        prepared_html = _prepare_html(text, input.max_characters)

    system_instruction = (
        "You are an expert research assistant that extracts the main readable content from "
        "web pages. Given raw HTML, identify the central article or body content. Convert "
        "it to Markdown with appropriate headings, lists, tables, and code blocks. Return "
        "a concise JSON object with `title`, `content_markdown`, and an optional `summary` "
        "of no more than three sentences."
    )

    user_prompt = (
        f"URL: {input.url}\n\n"
        "Extract the main readable content from the following HTML:\n"
        f"{prepared_html}"
    )

    completion = await generate(
        openai=openai,
        response_model=ReadWebsiteResultFromLLM,
        system_prompt=system_instruction,
        user_prompt=user_prompt,
    )

    return ReadWebsiteResult(
        url=input.url,
        title=completion.title,
        content_markdown=completion.content_markdown,
        summary=completion.summary,
    )


def _prepare_html(raw_html: str, max_characters: int) -> str:
    without_scripts = re.sub(
        r"<script.*?>.*?</script>", "", raw_html, flags=re.DOTALL | re.IGNORECASE
    )
    without_styles = re.sub(
        r"<style.*?>.*?</style>", "", without_scripts, flags=re.DOTALL | re.IGNORECASE
    )
    condensed = re.sub(r"\s+", " ", without_styles).strip()

    if len(condensed) > max_characters:
        return condensed[:max_characters]

    return condensed
