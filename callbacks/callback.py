

class Callback:
    def __init__(self, manager):
        self.manager = manager
    
    def on_fit_begin(self):
        pass
    
    def on_fit_end(self):
        pass
    
    def on_boosting_round_begin(self):
        pass

    def on_boosting_round_end(self):
        pass
