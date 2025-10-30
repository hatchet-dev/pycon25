from typing import TypeVar

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def response_to_pydantic(completion: ChatCompletion, model: type[T]) -> T:
    content = completion.choices[0].message.content

    if not content:
        raise TypeError("OpenAI returned empty content.")

    return model.model_validate_json(content)
