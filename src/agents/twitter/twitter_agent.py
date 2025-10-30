from hatchet_sdk import DurableContext
from pydantic import BaseModel

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
    message: str


class TwitterAgentOutput(BaseModel):
    tweet: str
    hashtags: list[str]


@hatchet.durable_task(name="twitter.twitter_agent", input_validator=TwitterAgentInput)
async def twitter_agent(
    input: TwitterAgentInput, ctx: DurableContext
) -> TwitterAgentOutput | None:
    ctx.log(f"Twitter agent received input: {input}")
    previous_tweet: str | None = None
    previous_feedback: str | None = None

    for _ in range(3):
        tweet = await compose_tweet.aio_run(
            input=ComposeTweetInput(
                prompt=input.message,
                previous_feedback=previous_feedback,
                previous_tweet=previous_tweet,
            )
        )
        judge_tweet_result = await judge_tweet.aio_run(
            input=JudgeTweetInput(tweet=tweet.tweet)
        )

        previous_feedback = judge_tweet_result.feedback
        previous_tweet = tweet.tweet

        if judge_tweet_result.should_publish:
            # await ctx.aio_wait_for("tweet:approved", )

            return TwitterAgentOutput(tweet=tweet.tweet, hashtags=tweet.hashtags)

    raise ValueError("Failed to generate a tweet")
