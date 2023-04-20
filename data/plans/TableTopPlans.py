{
    "Put all blocks on the bottom side of the table": """
#Get all blocks
objects = get_object_names()
blocks = [obj for obj in objects if 'block' in obj]

#Stack all blocks on the bottom side
for i, block in enumerate(blocks):
    put_first_on_second(block, 'bottom side')
""",
    "Put each block in a bowl of another color": """
#Get all blocks and all bowls
objects = get_object_names()
blocks = [obj for obj in objects if 'block' in obj]
bowls = [obj for obj in objects if 'bowl' in obj]

#For each block, put it in a bowl of a different color
for i, block in enumerate(blocks):
    #Get the color of the block
    block_color = block.split(' ')[0]
    
    #Find the bowl with a different color
    for bowl in bowls:
        bowl_color = bowl.split(' ')[0]
        if bowl_color != block_color:
            #Place the block in the bowl
            put_first_on_second(block, bowl)
            break
""",
    "Put all blocks inside the 'pink bowl'": """
# Put all the blocks inside the 'pink bowl'
put_first_on_second(blocks[0], 'pink bowl')
for block in blocks[1:]:
    put_first_on_second(block, blocks[0])""",
    "Put all blocks in a matching bowl": """
# Get all the bowls
objects = get_object_names()
bowls = [obj for obj in objects if 'bowl' in obj]

# Iterate through the blocks and put each block in the matching color bowl
for block in blocks:
    color = block.split(' ')[0]
    for bowl in bowls:
        if color in bowl:
            put_first_on_second(block, bowl)
            break""",
    "Pick up the block below the 'red block' and place it on cornor closest to 'pink bowl'": """
# Pick up the block below the 'red block'
target_block = find_object_related_to_obj('below', 'red block')
put_first_on_second(target_block, 'holder')

# Place the block on the corner closest to 'pink bowl'
target_corner = find_cornor_related_to_obj('closest to', 'pink bowl')
put_first_on_second(target_block, target_corner)""",
    "Pick up the block in the 'green bowl' and place it on the 'bottom right cornor' of the table": """
put_first_on_second('green block', 'bottom right corner')""",
    "Stack all blocks on the 'top right cornor'": """
#Get all blocks on table
objects = get_object_names()
blocks = [obj for obj in objects if 'block' in obj]

#Put the first block on the top right corner
put_first_on_second(blocks[0], 'top right corner')

#Stack all the remaining blocks on top of the first block
for i,block in enumerate(blocks[1:]):
    put_first_on_second(block, blocks[i-1])
""",
    "Pick up the 'green block' and place it in the corner closest to the 'yellow bowl'": """
#Pick up the 'green block' and place it in the corner closest to the 'yellow bowl'
put_first_on_second('green block', find_cornor_related_to_obj('closest to', 'yellow bowl'))""",
    "Put each block in a different cornor of the table": """

# Get all blocks on table
objects = get_object_names()
blocks = [obj for obj in objects if 'block' in obj]

# Get all table corners
corners = ['top left corner', 'top right corner', 'bottom left corner', 'bottom right corner']

# Assign each block to a corner using the modulo operator
for i, block in enumerate(blocks):
    corner_index = i % len(corners)
    target_corner = corners[corner_index]
    put_first_on_second(block, target_corner)
    """,
}


from control_actions import put_first_on_second
from plan_actions import get_object_names
from reasoning_actions import find_object_related_to_obj,  find_cornor_related_to_obj

#Write a code to put each block in a bowl of another color

#Put each block in a bowl of another color

#Get all blocks and all bowls
objects = get_object_names()
blocks = [obj for obj in objects if 'block' in obj]
bowls = [obj for obj in objects if 'bowl' in obj]

#For each block, put it in a bowl of a different color
for i, block in enumerate(blocks):
    #Get the color of the block
    block_color = block.split(' ')[0]
    
    #Find the bowl with a different color
    for bowl in bowls:
        bowl_color = bowl.split(' ')[0]
        if bowl_color != block_color:
            #Place the block in the bowl
            put_first_on_second(block, bowl)
            break