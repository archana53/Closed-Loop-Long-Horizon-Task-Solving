import argparse

import openai
from modules.planner.tasks import TableTopPickPlace, Task


class LLMPlanner(object):
    def __init__(self, task, params):
        self.task = task()
        self.api_key = params["key"]
        self.overwrite_cache = params["overwrite_cache"]
        self.scene_descriptor = ""
        if self.overwrite_cache:
            self.LLM_CACHE = {}
        else:
            self.LLM_CACHE = None

        openai.api_key = self.api_key

    def update_scene_descriptor(self, descriptor):
        self.scene_descriptor = descriptor

    def code_to_steps(codeplan: str):
        pass

    def generate_codeplan(self, instruction: str):
        task = instruction.replace(" ", "_")
        task += "():"
        prompt = (
            self.task.get_actions()
            + self.scene_descriptor
            + self.task.get_samples()
            + task
        )
        response = self.gpt3_call(prompt=prompt)
        codeplan = task + response["choices"][0]["text"]
        print(codeplan)
        return codeplan

    def gpt3_call(
        self,
        engine="text-davinci-002",
        prompt="",
        max_tokens=256,
        temperature=0,
        logprobs=1,
        echo=False,
    ):
        full_query = ""
        for p in prompt:
            full_query += p
        id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
        if id in self.LLM_CACHE.keys():
            response = self.LLM_CACHE[id]
        else:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                echo=echo,
            )
            self.LLM_CACHE[id] = response
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", type=str)
    parser.add_argument("--task", type=str, default="TableTopPickPlace")
    parser.add_argument("--cache", action="store_true", default=True)

    args = parser.parse_args()
    params = {"key": args.openai_key, "overwrite_cache": args.cache}
    our_planner = LLMPlanner(TableTopPickPlace, params)
    instruction = "stack all blocks on blue block"
    objects = (
        """\n objects = ["red block", "blue block", "yellow_block", "green_block"] \n"""
    )
    our_planner.update_scene_descriptor(objects)
    our_planner.generate_codeplan(instruction)
