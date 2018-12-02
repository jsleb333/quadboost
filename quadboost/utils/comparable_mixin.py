

class ComparableMixin:
    """
    Mixin class that delegates the rich comparison operators to the specified attribute.

    Note: Uses __init_subclass__ as a work around for a bug with the 'Queue' class of 'multiprocessing' when parallelizing.
    """
    def __init_subclass__(cls, *, cmp_attr):
        def get_cmp_attr(self): return getattr(self, cmp_attr)
        cls.cmp_attr = property(get_cmp_attr)

        for operator_name in ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__']:
            def operator_func(self, other, operator_name=operator_name):
                other_attr = other.cmp_attr if hasattr(other, 'cmp_attr') else other
                try:
                    return getattr(self.cmp_attr, operator_name)(other_attr)
                except TypeError:
                    return NotImplemented

            setattr(cls, operator_name, operator_func)
