"""
Comparison of 5 ways to implement a cloner inside a class. A cloner is an emulation of a constructor at call time, with the same initialization parameters. Hence it is not the same as a simple copy.

The idea is to intercept the arguments given at initialisation and to assign to the __call__ function a constructor.
"""

"""
Method 1: Using function decorators wrapped on the init

Advantages:
    - Simple
    - No need for additional structure to save the init args
    - Doc properly displayed
Disadvantages:
    - Decorates the __init__ method, but modifies other methods of the class
"""
from functools import wraps, update_wrapper

def call_to_clone(init_func):
    @wraps(init_func)
    def wrapper(self, *args, **kwargs):

        def clone(self):
            return type(self)(*args, **kwargs)
        type(self).__call__ = clone
        
        return init_func(self, *args, **kwargs)
    return wrapper

"""
Method 2: Using a metaclass acting as a decorator

Advantages:
    - Easy to read and understand
    - Seems to func in the purpose of metaclasses
    - Doc properly displayed
Disadvantages:
    - Possible conflict with other metaclasses when inherited
    - Stores the init args inside the class
"""
class MetaCloner(type):
    def __init__(cls, name, bases, attrs):
        cls.__init__ = cls.saved_init(cls.__init__)
        cls.__call__ = cls.clone
    
    def saved_init(cls, init_func):
        @wraps(init_func)
        def new_init(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs
            return init_func(self, *args, **kwargs)
        return new_init
    
    def clone(cls):
        return cls(*cls.init_args, **cls.init_kwargs)

"""
Method 3: Using a class decorator

Advantages:
    - Class makes things cleaner by separating the work into methods
    - Does not overload the __init__ method
Disadvantages:
    - Must stores the decorated class and init args
    - Decorator class construction is somewhat unconventional (define the __call__ method)
    - Doc not always properly displayed
"""
class cloner3:
    """
    cloner3 doc
    """
    def __init__(self, decorated_cls):
        """
        cloner3 init doc
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

"""
Method 4: Using function decorators to inherit the class

Advantages:
    - Power of a class with syntax of a class
Disadvantages:
    - Object returned is a new class that inherits the original
    - Doc not always properly displayed
"""
def cloner4(cls):
    """
    cloner4 doc
    """
    class _Cloner(cls):
        """
        cloner4 init doc
        """
        @wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            super().__init__(*args, **kwargs)
            update_wrapper(self, cls)
        
        def __call__(self):
            return cls(*self.args, **self.kwargs)
    return _Cloner

"""
Method 5: Using function decorators to modify the class

Advantages:
    - Simple
    - No need for additional structure to save the init args
    - Does not overload the __init__ methods comparatively to other methods.
Disadvantages:
    - Doc not always properly displayed
"""
def cloner5(cls):
    @wraps(cls)
    def wrapper(*args, **kwargs):
        def clone(self):
            return cls(*args, **kwargs)
        cls.__call__ = clone
        return cls(*args, **kwargs)
    return wrapper

# @cloner4
# @cloner3
# @cloner5
# class Dummy:
class Dummy(metaclass=MetaCloner):
    """
    class doc
    """
    # @call_to_clone
    def __init__(self, a, *args, b=None, **kwargs):
        """
        init doc
        """
        self.a = a
        self.b = b
        print(a, b)
    
    def func(self):
        """
        func doc
        """
        print('a')


def main():
    c = Dummy(2, 3, b='test', x='autre')
    c.a = 'adad'
    d = c()
    d.func()
    
    
if __name__ == '__main__':
    # Dummy()
    main()