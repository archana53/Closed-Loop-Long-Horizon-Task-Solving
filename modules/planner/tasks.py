class TableTopPickPlace(object):
    def __init__(self):
        self.actions = ["pick <obj>", "place <obj> <loc>", "place <obj> <obj>"]
        self.samples =  [
        """
        def place_red_block_on_blue_block():
        #1 check whether blue block does not have anything on top
        assert('blue block' is 'free')
            else : pick(block on blue block)
                place(next to blue block)
        #2 pick red block
            pick('red block')

        #3 place on the blue block
            place('red block','blue block')

        #4 done()
        """
        ]
    def get_descriptor():
        pass
