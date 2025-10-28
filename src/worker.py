from hatchet_client import hatchet
from agents.researcher.tools.read_website import read_website
import agents.linkedin.tools as linkedin_tools
import agents.twitter.tools as twitter_tools


def main() -> None:
    worker = hatchet.worker(
        "test-worker", slots=1, workflows=[
            read_website,
            linkedin_tools.create_linkedin_post,
            linkedin_tools.simulate_linkedin_post,
            twitter_tools.compose_tweet,
        ])
    worker.start()


if __name__ == "__main__":
    main()
