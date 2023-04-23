{
"Stack all blocks on red block" :
"""from control_actions import put_first_on_second
from plan_actions import get_object_names, tell_planner
from reasoning_actions import find_object_related_to_obj

# List to track executed control actions
executed_actions = []

def validate_previous_actions():
    invalid_actions = []
    for index, action in enumerate(executed_actions):
        if action['type'] == 'put_first_on_second':
            obj_a = action['obj_a']
            obj_b = action['obj_b']
            current_obj_below = find_object_related_to_obj('below', obj_a)
            if current_obj_below != obj_b:
                invalid_actions.append((index, action))
    return invalid_actions

# Get all objects on table
objects = get_object_names()

# Filter out the blocks
blocks = [obj for obj in objects if 'block' in obj]

# Find the red block
red_block = None
for block in blocks:
    if 'red block' in block:
        red_block = block
        break

# Check if red block is present
if red_block is None:
    tell_planner("No red block found.")
else:
    # Remove red block from blocks list
    blocks.remove(red_block)
    # Start stacking on red block
    current_block = red_block

    while len(blocks) > 0:
        # Stack the next block on the current_block
        next_block = blocks.pop(0)

        while True:
            # Check if previous actions are still valid
            invalid_actions = validate_previous_actions()

            if not invalid_actions:
                # Stack the next block on the current_block
                put_first_on_second(next_block, current_block)
                executed_actions.append({
                    'type': 'put_first_on_second',
                    'obj_a': next_block,
                    'obj_b': current_block
                })
                print(f"Stacked the {next_block} on the {current_block}.")

                # Update current_block
                current_block = next_block
                break
            else:
                for index, invalid_action in invalid_actions:
                    tell_planner(f"Validation failed. Action {index}: {invalid_action['type']}({invalid_action['obj_a']}, {invalid_action['obj_b']}) is no longer valid. Redoing the action.")
                    # Redo the invalid action
                    put_first_on_second(invalid_action['obj_a'], invalid_action['obj_b'])

    tell_planner("Plan Completed.")

""",
"""Put all blocks in matching bowls""": """from control_actions import put_first_on_second
from plan_actions import get_object_names, tell_planner
from reasoning_actions import find_object_related_to_obj

# List to track executed control actions
executed_actions = []

def validate_previous_actions():
    invalid_actions = []
    for index, action in enumerate(executed_actions):
        if action['type'] == 'put_first_on_second':
            obj_a = action['obj_a']
            obj_b = action['obj_b']
            current_obj_below = find_object_related_to_obj('below', obj_a)
            if current_obj_below != obj_b:
                invalid_actions.append((index, action))
    return invalid_actions

# Get all objects on table
objects = get_object_names()

# Filter out the blocks and bowls
blocks = [obj for obj in objects if 'block' in obj]
bowls = [obj for obj in objects if 'bowl' in obj]

# Iterate through each block and find the matching colored bowl
for block in blocks:
    block_color = block.split(' ')[0]
    matching_bowl = None

    # Find the bowl with the same color as the block
    for bowl in bowls:
        bowl_color = bowl.split(' ')[0]
        if block_color == bowl_color:
            matching_bowl = bowl
            break

    while True:
        # Check if previous actions are still valid
        invalid_actions = validate_previous_actions()

        if not invalid_actions:
            # If a matching colored bowl is found, put the block in the bowl
            if matching_bowl is not None:
                put_first_on_second(block, matching_bowl)
                executed_actions.append({
                    'type': 'put_first_on_second',
                    'obj_a': block,
                    'obj_b': matching_bowl
                })
                print(f"Put the {block} in the {matching_bowl}.")
            else:
                tell_planner(f"No matching colored bowl found for the {block}.")
            break
        else:
            for index, invalid_action in invalid_actions:
                tell_planner(f"Validation failed. Action {index}: {invalid_action['type']}({invalid_action['obj_a']}, {invalid_action['obj_b']}) is no longer valid. Redoing the action.")
                # Redo the invalid action
                put_first_on_second(invalid_action['obj_a'], invalid_action['obj_b'])

tell_planner("Plan Completed.")
""",
"""Put all blocks in a mismatched bowl""":"""from control_actions import put_first_on_second
from plan_actions import get_object_names, tell_planner
from reasoning_actions import find_object_related_to_obj

# List to track executed control actions
executed_actions = []

def validate_previous_actions():
    invalid_actions = []
    for index, action in enumerate(executed_actions):
        if action['type'] == 'put_first_on_second':
            obj_a = action['obj_a']
            obj_b = action['obj_b']
            current_obj_below = find_object_related_to_obj('below', obj_a)
            if current_obj_below != obj_b:
                invalid_actions.append((index, action))
    return invalid_actions

# Get all objects on table
objects = get_object_names()

# Filter out the blocks and bowls
blocks = [obj for obj in objects if 'block' in obj]
bowls = [obj for obj in objects if 'bowl' in obj]

# Function to find a mismatched colored bowl for a given block
def find_mismatched_bowl(block_color, remaining_bowls):
    for bowl in remaining_bowls:
        bowl_color = bowl.split(' ')[0]
        if block_color != bowl_color:
            return bowl
    return None

# Iterate through each block and find a mismatched colored bowl
for block in blocks:
    block_color = block.split(' ')[0]
    mismatched_bowl = find_mismatched_bowl(block_color, bowls)

    while True:
        # Check if previous actions are still valid
        invalid_actions = validate_previous_actions()

        if not invalid_actions:
            # If a mismatched colored bowl is found, put the block in the bowl
            if mismatched_bowl is not None:
                put_first_on_second(block, mismatched_bowl)
                executed_actions.append({
                    'type': 'put_first_on_second',
                    'obj_a': block,
                    'obj_b': mismatched_bowl
                })
                print(f"Put the {block} in the {mismatched_bowl}.")
                # Remove the bowl from the list of available bowls
                bowls.remove(mismatched_bowl)
            else:
                tell_planner(f"No mismatched colored bowl found for the {block}.")
            break
        else:
            for index, invalid_action in invalid_actions:
                tell_planner(f"Validation failed. Action {index}: {invalid_action['type']}({invalid_action['obj_a']}, {invalid_action['obj_b']}) is no longer valid. Redoing the action.")
                # Redo the invalid action
                put_first_on_second(invalid_action['obj_a'], invalid_action['obj_b'])

tell_planner("Plan Completed.")
""",
"""Put all blocks in pink bowl""":"""
from control_actions import put_first_on_second
from plan_actions import get_object_names, tell_planner
from reasoning_actions import find_object_related_to_obj

# List to track executed control actions
executed_actions = []

def validate_previous_actions():
    invalid_actions = []
    for index, action in enumerate(executed_actions):
        if action['type'] == 'put_first_on_second':
            obj_a = action['obj_a']
            obj_b = action['obj_b']
            current_obj_below = find_object_related_to_obj('below', obj_a)
            if current_obj_below != obj_b:
                invalid_actions.append((index, action))
    return invalid_actions

# Get all objects on table
objects = get_object_names()

# Filter out the blocks and bowls
blocks = [obj for obj in objects if 'block' in obj]
bowls = [obj for obj in objects if 'bowl' in obj]

# Find the pink bowl
pink_bowl = None
for bowl in bowls:
    if 'pink bowl' in bowl:
        pink_bowl = bowl
        break

# Check if pink bowl is present
if pink_bowl is None:
    tell_planner("No pink bowl found.")
else:
    # Iterate through each block and put it in the pink bowl
    for block in blocks:
        while True:
            # Check if previous actions are still valid
            invalid_actions = validate_previous_actions()

            if not invalid_actions:
                # Put the block in the pink bowl
                put_first_on_second(block, pink_bowl)
                executed_actions.append({
                    'type': 'put_first_on_second',
                    'obj_a': block,
                    'obj_b': pink_bowl
                })
                print(f"Put the {block} in the {pink_bowl}.")
                break
            else:
                for index, invalid_action in invalid_actions:
                    tell_planner(f"Validation failed. Action {index}: {invalid_action['type']}({invalid_action['obj_a']}, {invalid_action['obj_b']}) is no longer valid. Redoing the action.")
                    # Redo the invalid action
                    put_first_on_second(invalid_action['obj_a'], invalid_action['obj_b'])

    tell_planner("Plan Completed.")
"""
}