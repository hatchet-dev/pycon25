from hatchet_sdk import DurableContext
from pydantic import BaseModel

from agents.researcher.tools.read_website import ReadWebsiteResult
from agents.twitter.tools.compose_tweet import (
    ComposeTweetInput,
    compose_tweet,
)
from agents.twitter.tools.judge_tweet import (
    JudgeTweetInput,
    judge_tweet,
)
from hatchet_client import hatchet


class TwitterAgentInput(BaseModel):
    message: str | None = None
    researcher_result: ReadWebsiteResult | None = None


class TwitterAgentOutput(BaseModel):
    tweet: str
    hashtags: list[str]


@hatchet.durable_task(name="twitter.twitter_agent")
async def twitter_agent(input: TwitterAgentInput, ctx: DurableContext) -> None:
    ctx.log(f"Twitter agent received input: {input}")

    for _ in range(3):
        tweet = await compose_tweet.aio_run(
            input=ComposeTweetInput(
                message=input.message,
                tone="punchy",
                include_hashtags=True,
                model="gpt-4o-mini",
                temperature=0.8,
            )
        )
        judge_tweet_result = await judge_tweet.aio_run(
            input=JudgeTweetInput(
                tweet=tweet.tweet, model="gpt-4o-mini", temperature=0.2
            )
        )
        if judge_tweet_result.should_publish:
            # await ctx.aio_wait_for("tweet:approved", )

            return TwitterAgentOutput(tweet=tweet.tweet, hashtags=tweet.hashtags)

    raise ValueError("Failed to generate a tweet")
