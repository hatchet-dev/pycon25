from typing import Annotated

from hatchet_sdk import Depends
from openai import AsyncOpenAI


async def openai_client() -> AsyncOpenAI:
    return AsyncOpenAI()


OpenAIDependency = Annotated[AsyncOpenAI, Depends(openai_client)]
