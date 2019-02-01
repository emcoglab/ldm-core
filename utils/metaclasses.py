"""
===========================
Metaclasses.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""


class Singleton(type):
    """
    Use as a metaclass to make a class a singleton.
    With thanks to https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python.
    """
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class UnsettableSingleton(Singleton):

    def _unset(cls):
        del cls._instances[cls]
