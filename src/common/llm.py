from openai import AsyncOpenAI

from .response import T, response_to_pydantic


async def generate(
    openai: AsyncOpenAI, response_model: type[T], system_prompt: str, user_prompt: str
) -> T:
    completion = await openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.80,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
            },
        },
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response_to_pydantic(completion, response_model)
