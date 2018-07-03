import functools


def cloner(cls):
    """
    This function decorator makes any weak learners clonable by setting the __call__ function as a constructor using the initialization parameters.
    """
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        def clone(self):
            return cls(*args, **kwargs)
        cls.__call__ = clone
        return cls(*args, **kwargs)
    return wrapper