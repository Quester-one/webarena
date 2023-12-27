"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any

import aiolimiter
import openai
import openai.error
import google.generativeai as genai
from tqdm.asyncio import tqdm_asyncio


def retry_with_exponential_backoff(  # type: ignore
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 3,
        errors: tuple[Any] = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


# TODO：add async


def remove_key_from_dicts_list(original_list, key_to_remove):
    new_list = [{k: v for k, v in my_dict.items() if k != key_to_remove} for my_dict in original_list]
    return new_list


def flatten_tuple_list(original_list):
    new_list = []

    for item in original_list:
        if isinstance(item, tuple):
            new_list.extend(item)
        else:
            new_list.append(item)

    return new_list


@retry_with_exponential_backoff
def generate_from_google_chat_completion(
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        context_length: int,
        stop_token: str | None = None,
) -> str:
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError(
            "GEMINI_API_KEY environment variable must be set when using GEMINI API."
        )
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    gemini_model = genai.GenerativeModel(model_name=model)
    messages = remove_key_from_dicts_list(messages, "name")
    # messages.insert(1, {"role": "model", "parts": [""]})
    new_messages = []
    for message in messages:
        new_messages.append(message["parts"][0])
    new_messages = flatten_tuple_list(new_messages)
    response = gemini_model.generate_content(contents=new_messages,
                                             generation_config=genai.types.GenerationConfig(
                                                 temperature=temperature,
                                                 max_output_tokens=max_tokens,
                                                 top_p=top_p))

    answer = response.text
    return answer
