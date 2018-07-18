

class Cloner:
    """
    This constructor class makes any weak learners clonable by setting the __call__ function as a constructor using the initialization parameters.
    This class, to be inherited, acts like a decorator around the constructor, but without the inconveniences of decorators (with pickling).
    """
    def __new__(cls, *args, **kwargs):
        def clone(self): return cls(*args, **kwargs)
        cls.__call__ = clone
        return super().__new__(cls)