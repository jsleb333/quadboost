"""
Comparison of 3 ways to implement a cloner inside a class. A cloner is an emulation of a constructor at call time, with the same initialization parameters. Hence it is NOT the same as a simple copy.

The idea is to intercept the arguments given at initialisation and to assign to the __call__ function a constructor.
"""

# Using function decorators
from functools import wraps
from functools import update_wrapper


def call_to_clone(init_func):
    @wraps(init_func)
    def wrapper(self, *args, **kwargs):

        def clone(self):
            return type(self)(*args, **kwargs)
        type(self).__call__ = clone
        
        return init_func(self, *args, **kwargs)
    return wrapper

class DummyFuncDecorator:
    """
    class doc string
    """
    @call_to_clone
    def __init__(self, a, *args, b=None, **kwargs):
        """
        init doc str
        """
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
    """
    cloner class doc str
    """
    def __init__(self, decorated_cls):
        """
        cloner init doc str
        """
        self.decorated_cls = decorated_cls
        self.decorated_cls.__call__ = self.clone
        update_wrapper(self, decorated_cls)
    
    def __call__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        return self.decorated_cls(*args, **kwargs)

    def clone(self):
        return self.decorated_cls(*self.init_args, **self.init_kwargs)

@Cloner
class DummyDecoratedClass:
    """
    class doc string
    """
    def __init__(self, a, *args, b=None, **kwargs):
        """
        init doc string
        """
        self.a = a
        self.b = b
        print(a, b)
    
    def func(self):
        """
        fit doc str
        """
        print('a')


def main():
    c = DummyDecoratedClass(2, 3, b='test', x='autre')
    c.a = 'adad'
    d = c()
    d.fit()

if __name__ == '__main__':
    main()


