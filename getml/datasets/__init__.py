"""
The :mod:`getml.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets. It also
features some artificial data generators.
"""

from .samples_generator import make_numerical
from .samples_generator import make_categorical
from .samples_generator import make_discrete
from .samples_generator import make_same_units_categorical
from .samples_generator import make_same_units_numerical
from .samples_generator import make_snowflake
