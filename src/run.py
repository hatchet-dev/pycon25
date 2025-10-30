import asyncio

from agents.twitter.twitter_agent import TwitterAgentInput, twitter_agent


async def main() -> None:
    result = await twitter_agent.aio_run(
        input=TwitterAgentInput(
            message="Post something fun about our PyCon workshop for the world to see!"
        )
    )
    print(result)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
