

class Callback:
    """
    Simple Callback class. A Callback implements 4 methods: 'on_iteration_begin', 'on_iteration_end', 'on_step_begin' and 'on_step_end'. These methods will be called at the appropriate time by their 'manager'.
    """
    def __init__(self, manager=None):
        """
        Args:
            manager (Reference to the CallbacksManagerIterator, optional): The manager can be set in the constructor or can be set later. The manager should not let to None, since most callbacks use this reference.
        """
        self.manager = manager

    def on_iteration_begin(self):
        pass

    def on_iteration_end(self):
        pass

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass

    def on_exception_exit(self, exception_type=None, exception_message=None, traceback=None):
        pass
