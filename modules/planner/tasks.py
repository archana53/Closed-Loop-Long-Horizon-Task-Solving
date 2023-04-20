from data.prompts.TableTopManipulation import (
    Prompt as TableTopManipulationPrompt,
)

class Task(object):
    def __init__(self):
        self.actions = None
        self.samples = None


class TableTopPickPlace(Task):
    def __init__(self):
        super(TableTopPickPlace, self).__init__()

        self.actions = [
            "pick",
            "place",
        ]
        self.samples = [
            """def stack_all_remaining_blocks_on_red_block():
    #Identify the current block to stack on
    object_to_stack_on = 'red_block'
    for i, current_object in enumerate(objects):
        if (current_object != 'red_block'):
            #Pick current object
            pick(current_object)

            #Place on the object on top
            place(current_object, object_to_stack_on)
            
            #Update the current block to stack on to the object on top
            object_to_stack_on = current_object
    #Done
        """,
        ]

    def get_actions(self):
        actions_string = "from actions import "
        actions_string += ",".join(self.actions)
        return actions_string

    def get_samples(self):
        samples_string = "\n".join(self.samples)
        return samples_string

    def get_prompt(self):
        return TableTopManipulationPrompt
