

class Callback:
    def __init__(self, manager=None):
        self.manager = manager

    def on_iteration_begin(self):
        pass

    def on_iteration_end(self):
        pass

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass
