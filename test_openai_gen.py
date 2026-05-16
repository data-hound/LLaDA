#!/usr/bin/env python

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from eval_kakuro_openai import run_openai_generation_one

load_dotenv()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_openai_gen.py <prompt>")
        sys.exit(1)
    
    token = os.getenv("GEMINI_API_KEY")
    # token = os.getenv("LITELLM_TOKEN")
    # token = os.getenv("OPENAI_API_TOKEN")
    if not token:
        print("Please set the LITELLM_TOKEN environment variable.")
        sys.exit(1)

    prompt = sys.argv[1]
    # client = OpenAI(
    #     api_key=token,
    #     base_url="https://litellm.oit.duke.edu/v1",
    # )
    client = OpenAI(
                    api_key=token,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )

    # response = client.responses.create(
    #     model="GPT 4.1",
    #     instructions="You are a helpful assistant here to demo the power of AI",
    #     input=prompt,
    # )
    models = client.models.list()
    print("Available Models:")
    for model in models:
        print(model.id)

    # response = run_openai_generation_one(
    #                                         client=client, 
    #                                         model_name="Mistral on-site", 
    #                                         prompt=prompt,
    #                                         max_new_tokens=512,
    #                                         temperature=0.1,
    #                                         retries=2,
    #                                         retry_sleep=100
    #                                         )

    # # print(response.output[0].content[0].text)
    # print(response)