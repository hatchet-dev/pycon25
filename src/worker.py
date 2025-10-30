from agents.linkedin.tools.create_post import create_linkedin_post
from agents.linkedin.tools.simulate_post import simulate_linkedin_post
from agents.researcher.tools.read_website import read_website
from agents.twitter.tools.compose_tweet import compose_tweet
from agents.twitter.tools.judge_tweet import judge_tweet
from agents.twitter.twitter_agent import twitter_agent
from hatchet_client import hatchet


def main() -> None:
    worker = hatchet.worker(
        "agent-worker",
        workflows=[
            read_website,
            create_linkedin_post,
            simulate_linkedin_post,
            compose_tweet,
            judge_tweet,
            twitter_agent,
        ],
    )
    worker.start()


if __name__ == "__main__":
    main()
