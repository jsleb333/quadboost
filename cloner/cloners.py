"""
Comparison of 3 ways to implement a cloner inside a class. A cloner is an emulation of a constructor at call time, with the same initialization parameters. Hence it is NOT the same as a simple copy.

The idea is to intercept the arguments given at initialisation and to assign to the __call__ function a constructor.
"""

# Using function decorators
from functools import wraps

def call_to_clone(init_func):
    @wraps(init_func)
    def wrapper(self, *args, **kwargs):

        def clone(self):
            return type(self)(*args, **kwargs)
        type(self).__call__ = clone
        
        return init_func(self, *args, **kwargs)
    return wrapper

class DummyFuncDecorator:
    @call_to_clone
    def __init__(self, a, *args, b=None, **kwargs):
        self.a = a
        self.b = b
        print(a, b)


# Using a metaclass acting as a decorator
class MetaCloner(type):
    def __init__(cls, name, bases, attrs):
        cls.__init__ = cls.saved_init(cls.__init__)
        cls.__call__ = cls.clone
    
    def saved_init(cls, init_func):
        @wraps(init_func)
        def new_init(self, *args, **kwargs):
            cls.init_args = args
            cls.init_kwargs = kwargs
            return init_func(self, *args, **kwargs)
        return new_init
    
    def clone(cls):
        return cls(*cls.init_args, **cls.init_kwargs)

class DummyMetaclass(metaclass=MetaCloner):
    def __init__(self, a, *args, b=None, **kwargs):
        self.a = a
        self.b = b
        print(a, b)
    

# Using a class decorator
class Cloner:
    def __init__(self, decorated_cls):
        self.decorated_cls = decorated_cls
        decorated_cls.__call__ = self.clone

    def clone(self):
        return self.decorated_cls(*self.init_args, **self.init_kwargs)
    
    def __call__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        return self.decorated_cls(*args, **kwargs)

@Cloner
class DummyClassDecorator:
    def __init__(self, a, *args, b=None, **kwargs):
        self.a = a
        self.b = b
        print(a, b)


def main():
    c = DummyMetaclass(2, 3, b='test', x='autre')
    c.a = 'adad'
    d = c()

if __name__ == '__main__':
    main()


