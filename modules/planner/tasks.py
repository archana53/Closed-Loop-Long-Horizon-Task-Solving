class Task(object):
    def __init__(self):
        self.actions = None
        self.samples = None


class TableTopPickPlace(Task):
    def __init__(self):
        super(TableTopPickPlace, self).__init__()

        self.actions = ["pick <obj>", "place <obj> <loc>", "place <obj> <obj>"]
        self.samples = [
            """
        def place_red_block_on_blue_block():
        #1 check whether blue block does not have anything on top
        assert('blue block has nothing on top')

        #2 pick red block
            pick('red block')

        #3 place on the blue block
            place('red block','blue block')

        #4 done()
        """
        ]

    def get_actions(self):
        actions_string = "from actions import "
        actions_string += ",".join(self.actions)
        return actions_string

    def get_samples(self):
        samples_string = "\n".join(self.samples)
        return samples_string
