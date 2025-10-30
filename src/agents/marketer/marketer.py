from hatchet_sdk import Context
from pydantic import BaseModel

from hatchet_client import hatchet


class MarketerInput(BaseModel):
    message: str


@hatchet.durable_task(name="marketer.marketer", input_validator=MarketerInput)
async def marketer(input: MarketerInput, ctx: Context) -> None:
    ## todo: fill this in
    ## top level marketer agent entrypoint, should build off other agents
    return None

    # ctx.log(f"Marketer received message: {input.message}")

    # # first the marketer will delegate the task to the researcher to read the website
    # await read_website.aio_run(
    #     input=ReadWebsiteInput(
    #         prompt=input.message,
    #         tone="punchy",
    #         include_hashtags=True,
    #         model="gpt-4o-mini",
    #         temperature=0.8,
    #     ),
    # )

    # return input.message
