import openai
from cloper.planner.tasks import TableTopPickPlace


class LLMPlanner(object):
    def __init__(self, task, params):
        self.task = task
        self.api_key = params['key']
        self.overwrite_cache = params['overwrite_cache']
        self.scene_descriptor = ""
        if self.overwrite_cache:
            self.LLM_CACHE = {}
        else :
            self.LLM_CACHE = None
        
    def update_scene_descriptor(self, descriptor):
        self.scene_descriptor = descriptor

    def code_to_steps(codeplan : str):
        pass

    def generate_codeplan(self, instruction : str):
        current_task = instruction.replace(' ','_')
        current_task += '():'
        prompt = self.task.actions +  self.scene_descriptor + self.task.samples + current_task



    def gpt3_call(self, engine="text-davinci-002", prompt="", max_tokens=128, temperature=0, 
                logprobs=1, echo=False):
        full_query = ""
        for p in prompt:
            full_query += p
        id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
        if id in self.LLM_CACHE.keys():
            response = self.LLM_CACHE[id]
        else:
            response = openai.Completion.create(engine=engine, 
                                                prompt=prompt, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature,
                                                logprobs=logprobs,
                                                echo=echo)
            self.LLM_CACHE[id] = response
        return response
