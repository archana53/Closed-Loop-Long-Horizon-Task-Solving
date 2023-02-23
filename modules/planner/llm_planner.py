import openai


class LLMPlanner(object):
    def __init__(self, params):
        self.overwrite_cache = params['overwrite_cache']
        if self.overwrite_cache:
            self.LLM_CACHE = {}
        else :
            self.LLM_CACHE = None

    def get_scene_descriptor():
        pass

    def code_to_steps():
        pass

    def gpt3_call(self, engine="text-ada-001", prompt="", max_tokens=128, temperature=0, 
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
