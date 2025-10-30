"""Hatchet task for fetching and extracting readable content from a web page."""

from __future__ import annotations

import re

import requests
from hatchet_sdk import Context
from openai import OpenAI
from pydantic import BaseModel, Field, HttpUrl

from common.response import response_to_pydantic
from hatchet_client import hatchet

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_CHARACTERS = 20_000


class ReadWebsiteInput(BaseModel):
    """Validated payload for the ``read_website`` task."""

    url: HttpUrl = Field(..., description="URL of the web page to retrieve.")
    timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description="Timeout for the HTTP request in seconds.",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="OpenAI Responses model used to extract readable content.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature passed to the OpenAI model.",
    )
    max_characters: int = Field(
        default=DEFAULT_MAX_CHARACTERS,
        ge=1_000,
        le=100_000,
        description="Maximum number of HTML characters forwarded to the model.",
    )


class ReadWebsiteResultFromLLM(BaseModel):
    """Structured representation of extracted page content."""

    url: HttpUrl
    title: str
    content_markdown: str = Field(
        description="Primary page content converted to Markdown."
    )
    summary: str | None = Field(
        default=None, description="Optional short summary (≤3 sentences) of the page."
    )


class ReadWebsiteResult(ReadWebsiteResultFromLLM):
    """Structured representation of extracted page content."""

    url: HttpUrl


@hatchet.task(name="researcher.read-website", input_validator=ReadWebsiteInput)
def read_website(input: ReadWebsiteInput, ctx: Context) -> ReadWebsiteResult:
    """Fetch a web page and ask OpenAI to extract readable Markdown content."""

    ctx.log(
        f"Fetching URL `{input.url}` with timeout `{input.timeout_seconds}` seconds "
        f"and model `{input.model}`."
    )

    try:
        response = requests.get(str(input.url), timeout=input.timeout_seconds)
    except requests.RequestException as exc:  # pragma: no cover - network failures
        raise RuntimeError(f"Failed to fetch `{input.url}`: {exc}") from exc

    if response.status_code >= 400:
        raise RuntimeError(
            f"Request to `{input.url}` failed with status code {response.status_code}."
        )

    prepared_html = _prepare_html(response.text, input.max_characters)

    client = OpenAI()

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

    ai_response = client.chat.completions.create(
        model=input.model,
        temperature=input.temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ReadWebsiteResultFromLLM",
                "schema": ReadWebsiteResultFromLLM.model_json_schema(),
            },
        },
        messages=[
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    parsed = response_to_pydantic(ai_response, ReadWebsiteResultFromLLM)

    return ReadWebsiteResult(
        url=input.url,
        title=parsed.title,
        content_markdown=parsed.content_markdown,
        summary=parsed.summary,
    )


def _prepare_html(raw_html: str, max_characters: int) -> str:
    """Strip scripts/styles and truncate HTML before sending it to the model."""

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
