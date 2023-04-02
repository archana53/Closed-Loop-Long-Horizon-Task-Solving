import argparse

import openai
from modules.planner.tasks import TableTopPickPlace, Task
from data.prompts.TableTopManipulation import prompt


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

    def generate_codeplan(self, task: str):
        prompt = self.task.get_prompt()
        response = self.gpt3_call(prompt=prompt, task=instruction)
        codeplan = response["choices"][0]["message"]["content"]
        return codeplan

    def gpt3_call(
        self,
        model="gpt-3.5-turbo",
        prompt="",
        task="",
        max_tokens=256,
        temperature=0,
        logprobs=1,
        echo=False,
    ):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": self.prompt},
                {
                    "role": "user",
                    "content": "Based on the examples provided above, write python code to "
                    + task,
                },
            ],
        )
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
    objects = """objects = ["red block", "blue block", "yellow_block", "green_block"] \n"""
    our_planner.update_scene_descriptor(objects)
    our_planner.generate_codeplan(instruction)
