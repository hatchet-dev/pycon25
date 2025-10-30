from typing import Annotated, Any

from hatchet_sdk import Depends
from openai import AsyncOpenAI


async def openai_client(_i: Any, _c: Any) -> AsyncOpenAI:
    return AsyncOpenAI()


OpenAIDependency = Annotated[AsyncOpenAI, Depends(openai_client)]
