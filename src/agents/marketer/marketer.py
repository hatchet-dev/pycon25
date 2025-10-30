from hatchet_sdk import Context
from pydantic import BaseModel

from agents.researcher.tools.read_website import ReadWebsiteInput, read_website
from hatchet_client import hatchet


class MarketerInput(BaseModel):
    message: str


@hatchet.durable_task(name="marketer.marketer")
async def marketer(input: MarketerInput, ctx: Context) -> None:
    ctx.log(f"Marketer received message: {input.message}")

    # first the marketer will delegate the task to the researcher to read the website
    await read_website.aio_run(
        input=ReadWebsiteInput(
            prompt=input.message,
            tone="punchy",
            include_hashtags=True,
            model="gpt-4o-mini",
            temperature=0.8,
        ),
    )

    return input.message
