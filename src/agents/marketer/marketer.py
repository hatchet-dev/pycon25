from hatchet_sdk import Context
from pydantic import BaseModel

from agents.researcher.tools.read_website import read_website
from agents.twitter.tools.compose_tweet import ComposeTweetInput
from hatchet_client import hatchet


class MarketerInput(BaseModel):
    message: str


@hatchet.durable_task(name="marketer.marketer")
async def marketer(input: MarketerInput, ctx: Context) -> None:
    ctx.log(f"Marketer received message: {input.message}")

    # first the marketer will delegate the task to the researcher to read the website
    await read_website.aio_run(
        input=ComposeTweetInput(
            message=input.message,
            tone="punchy",
            include_hashtags=True,
            model="gpt-4o-mini",
            temperature=0.8,
        ),
    )

    return input.message
