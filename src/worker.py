import agents.linkedin.tools as linkedin_tools
from agents.researcher.tools.read_website import read_website
from agents.twitter.tools import compose_tweet, judge_tweet
from agents.twitter.twitter_agent import twitter_agent
from hatchet_client import hatchet


def main() -> None:
    worker = hatchet.worker(
        "agent-worker",
        workflows=[
            read_website,
            linkedin_tools.create_linkedin_post,
            linkedin_tools.simulate_linkedin_post,
            compose_tweet,
            judge_tweet,
            twitter_agent,
        ],
    )
    worker.start()


if __name__ == "__main__":
    main()
