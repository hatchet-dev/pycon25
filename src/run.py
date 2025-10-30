import asyncio

from agents.researcher.tools.read_website import read_website


async def main() -> None:
    result = await read_website.aio_run()
    print(result)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
