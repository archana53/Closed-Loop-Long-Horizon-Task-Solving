Prompt = """from control_actions import put_first_on_second
from plan_actions import get_object_names
from reasoning_actions import find_object_related_to_obj,  find_cornor_related_to_obj

#Get all blocks on table
objects = get_object_names()
blocks = [obj for obj in objects if 'block' in obj]

#Get color of the first block
color = blocks[0].split(' ')[0]
print(color)

# Pick up the 'blue block' and place it on the 'red bowl'
put_first_on_second("blue block", "red bowl")

#Get object on left of 'yellow bowl'
target_object =  find_object_related_to_obj('left of', 'yellow bowl')
print(target_object)

#Get object below the 'pink block' 
target_object =  find_object_related_to_obj('below', 'pink block')
print(target_object)

#Stack all the blocks
objects = get_object_names()
blocks = [obj for obj in objects if 'block' in obj]
for i, block in enumerate(blocks[1:]):
    put_first_on_second(block, blocks[i-1])

"""