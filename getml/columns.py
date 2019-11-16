# Copyright 2019 The SQLNet Company GmbH

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Columns are part of the :class:`~getml.engine.DataFrame`. This module contains routines for manipulating 
them.
"""

import copy
import json
import numbers

import numpy as np
import pandas as pd

import getml.communication as comm
import getml.container as container

# ------------------------------------------------------------------------------

class _Aggregation(object):
    def __init__(
        self,
        alias,
        col,
        type
    ):
        self.thisptr = dict()
        self.thisptr["as_"] = alias
        self.thisptr["col_"] = col.thisptr
        self.thisptr["type_"] = type

    # --------------------------------------------------------------------------

    def get(self):
        """
        Receives the value of the aggregation over the column.
        """

        # -------------------------------------------
        # Build command string

        cmd = dict()

        cmd["name_"] = ""
        cmd["type_"] = "Column.aggregate"

        cmd["aggregation_"] = self.thisptr
        cmd["df_name_"] = self.thisptr["col_"]["df_name_"]

        # -------------------------------------------
        # Create connection and send the command

        s = send_and_receive_socket(cmd)

        msg = comm.recv_string(s)

        # -------------------------------------------
        # Make sure everything went well, receive data
        # and close connection

        if msg != "Success!":
            s.close()
            raise Exception(msg)

        mat = comm.recv_matrix(s)

        # -------------------------------------------
        # Close connection.

        s.close()

        # -------------------------------------------

        return mat.ravel()[0]

# -----------------------------------------------------------------------------

class _VirtualBooleanColumn(object):

    def __init__(
        self,
        df_name,
        operator,
        operand1,
        operand2
    ):
        self.thisptr = dict()

        self.thisptr["df_name_"] = df_name

        self.thisptr["type_"] = "VirtualBooleanColumn"

        self.thisptr["operator_"] = operator

        self.thisptr["operand1_"] = self.__parse_operand(operand1)

        if operand2 is not None:
            self.thisptr["operand2_"] = self.__parse_operand(operand2)

    # -----------------------------------------------------------------------------

    def __and__(self, other):
        return _VirtualBooleanColumn(
            df_name=self.thisptr["df_name_"],
            operator="and",
            operand1=self,
            operand2=other
        )

    # -----------------------------------------------------------------------------

    def __eq__(self, other):
        return _VirtualBooleanColumn(
            df_name=self.thisptr["df_name_"],
            operator="equal_to",
            operand1=self,
            operand2=other
        )

    # -----------------------------------------------------------------------------

    def __or__(self, other):
        return _VirtualBooleanColumn(
            df_name=self.thisptr["df_name_"],
            operator="or",
            operand1=self,
            operand2=other
        )

    # -----------------------------------------------------------------------------

    def __ne__(self, other):
        return _VirtualBooleanColumn(
            df_name=self.thisptr["df_name_"],
            operator="not_equal_to",
            operand1=self,
            operand2=other
        )

    # -----------------------------------------------------------------------------

    def __xor__(self, other):
        return _VirtualBooleanColumn(
            df_name=self.thisptr["df_name_"],
            operator="xor",
            operand1=self,
            operand2=other
        )

    # -----------------------------------------------------------------------------

    def __parse_operand(self, operand):

        if isinstance(operand, bool):
            return {"type_": "BooleanValue", "value_": operand}

        elif isinstance(operand, str):
            return {"type_": "CategoricalValue", "value_": operand}

        elif isinstance(operand, numbers.Number):
            return {"type_": "Value", "value_": operand}

        else:
            if self.thisptr["operator_"] in ["and", "or", "not", "xor"]:
                if operand.thisptr["type_"] != "VirtualBooleanColumn":
                    raise Exception("This operator can only be applied to a BooleanColumn!")

            return operand.thisptr

    # -----------------------------------------------------------------------------

    def get(self):
        """
        Transform column to numpy array
        """

        # -------------------------------------------
        # Build command string

        cmd = dict()

        cmd["name_"] = self.thisptr["df_name_"]
        cmd["type_"] = "BooleanColumn.get"

        cmd["col_"] = self.thisptr

        # -------------------------------------------
        # Send command to engine

        s = comm.send_and_receive_socket(cmd)

        msg = comm.recv_string(s)

        # -------------------------------------------
        # Make sure everything went well, receive data
        # and close connection

        if msg != "Found!":
            s.close()
            raise Exception(msg)

        mat = comm.recv_boolean_matrix(s)

        # -------------------------------------------
        # Close connection, if necessary.

        s.close()

        # -------------------------------------------

        return mat.ravel()

    # -----------------------------------------------------------------------------

    def is_false(self):
        """Whether an entry is False - effectively inverts the Boolean column."""
        return _VirtualBooleanColumn(
            df_name=self.thisptr["df_name_"],
            operator="not",
            operand1=self,
            operand2=None
        )

# -----------------------------------------------------------------------------

class CategoricalColumn(container.Container):
    """
    Handle for categorical data that is kept in the getML engine

    Args:
        name (str): Name of the categorical column.
        role (str): Role that the column plays.
        num (int): Number of the column.
        unit (str): Unit of the column.
        df_name (str): Name of the DataFrame containing this column.
    """

    num_categorical_matrices = 0

    def __init__(
        self,
        name="",
        unit="",
        role="categorical",
        num=0,
        df_name=""
    ):

        super(CategoricalColumn, self).__init__()

        CategoricalColumn.num_categorical_matrices += 1
        if name == "":
            self.name = "CategoricalColumn " + \
                str(CategoricalColumn.num_categorical_matrices)
        else:
            self.name = name

        self.thisptr = dict()

        self.thisptr["df_name_"] = df_name

        self.thisptr["name_"] = self.name

        self.thisptr["role_"] = role

        self.thisptr["type_"] = "CategoricalColumn"

        self.thisptr["unit_"] = unit

# -----------------------------------------------------------------------------

class _VirtualCategoricalColumn(object):

    def __init__(
        self,
        df_name,
        operator,
        operand1,
        operand2
    ):
        self.thisptr = dict()

        self.thisptr["df_name_"] = df_name

        self.thisptr["type_"] = "VirtualCategoricalColumn"

        self.thisptr["operator_"] = operator

        if operand1 is not None:
            self.thisptr["operand1_"] = self.__parse_operand(operand1)

        if operand2 is not None:
            self.thisptr["operand2_"] = self.__parse_operand(operand2)

    # -----------------------------------------------------------------------------

    def __parse_operand(self, operand):

        if isinstance(operand, str):
            return {"type_": "CategoricalValue", "value_": operand}

        else:
            if self.thisptr["operator_"] != "to_str" and operand.thisptr["type_"] != "CategoricalColumn" and operand.thisptr["type_"] != "VirtualCategoricalColumn":
                raise Exception("This operator can only be applied to a CategoricalColumn!")

            if self.thisptr["operator_"] == "to_str" and\
                operand.thisptr["type_"] != "Column" and\
                operand.thisptr["type_"] != "VirtualColumn" and\
                operand.thisptr["type_"] != "VirtualBooleanColumn":
                raise Exception("This operator can only be applied to a Column!")

            return operand.thisptr

# -----------------------------------------------------------------------------


class Column(container.Container):
    """
    Handle for numerical data that is kept in the engine

    Args:
        name (str): Name of the categorical column.
        role (str): Role that the column plays.
        num (int): Number of the column.
        unit (str): Unit of the column.
        df_name (str): Name of the DataFrame containing this column.
    """

    num_columns = 0

    def __init__(
        self,
        name="",
        unit="",
        role="numerical",
        num=0,
        df_name=""
    ):

        super(Column, self).__init__()

        Column.num_columns += 1
        if name == "":
            self.name = "Column " + \
                str(Column.num_columns)
        else:
            self.name = name

        self.thisptr = dict()

        self.thisptr["df_name_"] = df_name

        self.thisptr["name_"] = self.name

        self.thisptr["role_"] = role

        self.thisptr["type_"] = "Column"

        self.thisptr["unit_"] = unit

# -----------------------------------------------------------------------------

class _VirtualColumn(object):

    def __init__(
        self,
        df_name,
        operator,
        operand1,
        operand2
    ):
        self.thisptr = dict()

        self.thisptr["df_name_"] = df_name

        self.thisptr["type_"] = "VirtualColumn"

        self.thisptr["operator_"] = operator

        if operand1 is not None:
            self.thisptr["operand1_"] = self.__parse_operand(operand1)

        if operand2 is not None:
            self.thisptr["operand2_"] = self.__parse_operand(operand2)

    # -----------------------------------------------------------------------------

    def __parse_operand(self, operand):

        if isinstance(operand, numbers.Number):
            return {"type_": "Value", "value_": operand}

        else:
            special_ops = ["to_num", "to_ts"]

            if self.thisptr["operator_"] not in special_ops and operand.thisptr["type_"] != "Column" and operand.thisptr["type_"] != "VirtualColumn":
                raise Exception("This operator can only be applied to a Column!")

            if self.thisptr["operator_"] in special_ops and operand.thisptr["type_"] != "CategoricalColumn" and operand.thisptr["type_"] != "VirtualCategoricalColumn":
                raise Exception("This operator can only be applied to a CategoricalColumn!")

            return operand.thisptr

# -----------------------------------------------------------------------------

def __abs(self):
    """Compute absolute value."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="abs",
        operand1=self,
        operand2=None
    )

Column.abs = __abs
_VirtualColumn.abs = __abs

# -----------------------------------------------------------------------------

def __acos(self):
    """Compute arc cosine."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="acos",
        operand1=self,
        operand2=None
    )

Column.acos = __acos
_VirtualColumn.acos = __acos

# -----------------------------------------------------------------------------

def __add(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="plus",
        operand1=self,
        operand2=other
    )

Column.__add__ = __add
Column.__radd__ = __add

_VirtualColumn.__add__ = __add
_VirtualColumn.__radd__ = __add

# -----------------------------------------------------------------------------

def __alias(self, alias):
    """
    Adds an alias to the column. This is useful for joins.

    Args:
        alias (str): The name of the column as it should appear in the new DataFrame.
    """
    col = copy.deepcopy(self)
    col.thisptr["as_"] = alias
    return col

CategoricalColumn.alias = __alias
Column.alias = __alias

# -----------------------------------------------------------------------------

def __assert_equal(self, alias):
    """
    ASSERT EQUAL aggregation.

    Throws an exception if not all values inserted
    into the aggregation are equal.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "assert_equal")

Column.assert_equal = __assert_equal
_VirtualColumn.assert_equal = __assert_equal


# -----------------------------------------------------------------------------

def __asin(self):
    """Compute arc sine."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="asin",
        operand1=self,
        operand2=None
    )

Column.asin = __asin
_VirtualColumn.asin = __asin

# -----------------------------------------------------------------------------

def __atan(self):
    """Compute arc tangent."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="atan",
        operand1=self,
        operand2=None
    )

Column.atan = __atan
_VirtualColumn.atan = __atan

# -----------------------------------------------------------------------------

def __avg(self, alias="new_column"):
    """
    AVG aggregation.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "avg")

Column.avg = __avg
_VirtualColumn.avg = __avg

# -----------------------------------------------------------------------------

def __cbrt(self):
    """Compute cube root."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="cbrt",
        operand1=self,
        operand2=None
    )

Column.cbrt = __cbrt
_VirtualColumn.cbrt = __cbrt


# -----------------------------------------------------------------------------

def __ceil(self):
    """Round up value."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="ceil",
        operand1=self,
        operand2=None
    )

Column.ceil = __ceil
_VirtualColumn.ceil = __ceil

# -----------------------------------------------------------------------------

def __concat(self, other):
    return _VirtualCategoricalColumn(
        df_name=self.thisptr["df_name_"],
        operator="concat",
        operand1=self,
        operand2=other
    )

def __rconcat(self, other):
    return _VirtualCategoricalColumn(
        df_name=self.thisptr["df_name_"],
        operator="concat",
        operand1=other,
        operand2=self
    )

CategoricalColumn.__add__ = __concat
CategoricalColumn.__radd__ = __rconcat

_VirtualCategoricalColumn.__add__ = __concat
_VirtualCategoricalColumn.__radd__ = __rconcat

# -----------------------------------------------------------------------------

def __contains(self, other):
    """
    Test containment

    Returns a boolean column indicating whether a
    string or column entry is contained in the corresponding
    entry of the other column.
    """
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="contains",
        operand1=self,
        operand2=other
    )

CategoricalColumn.contains = __contains

_VirtualCategoricalColumn.contains = __contains

# -----------------------------------------------------------------------------

def __cos(self):
    """Compute cosine."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="cos",
        operand1=self,
        operand2=None
    )

Column.cos = __cos
_VirtualColumn.cos = __cos

# -----------------------------------------------------------------------------

def __count(self, alias="new_column"):
    """
    COUNT aggregation.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "count")

Column.count = __count
_VirtualColumn.count = __count

# -----------------------------------------------------------------------------

def __count_categorical(self, alias="new_column"):
    """
    COUNT aggregation.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "count_categorical")


CategoricalColumn.count = __count_categorical
_VirtualCategoricalColumn.count = __count_categorical

# -----------------------------------------------------------------------------

def __count_distinct(self, alias="new_column"):
    """
    COUNT DISTINCT aggregation.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "count_distinct")

CategoricalColumn.count_distinct = __count_distinct
_VirtualCategoricalColumn.count_distinct = __count_distinct

# -----------------------------------------------------------------------------

def __day(self):
    """
    Extract day (of the month) from a time stamp.

    If the column is discrete or numerical, that number will be interpreted
    as the number of days since epoch time (January 1, 1970).
    """
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="day",
        operand1=self,
        operand2=None
    )

Column.day = __day
_VirtualColumn.day = __day

# -----------------------------------------------------------------------------

def __eq(self, other):
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="equal_to",
        operand1=self,
        operand2=other
    )

Column.__eq__ = __eq
_VirtualColumn.__eq__ = __eq

CategoricalColumn.__eq__ = __eq
_VirtualCategoricalColumn.__eq__ = __eq

# -----------------------------------------------------------------------------

def __erf(self):
    """Compute error function."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="erf",
        operand1=self,
        operand2=None
    )

Column.erf = __erf
_VirtualColumn.erf = __erf

# -----------------------------------------------------------------------------

def __exp(self):
    """Compute exponential function."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="exp",
        operand1=self,
        operand2=None
    )

Column.exp = __exp
_VirtualColumn.exp = __exp

# -----------------------------------------------------------------------------

def __floor(self):
    """Round down value."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="floor",
        operand1=self,
        operand2=None
    )

Column.floor = __floor
_VirtualColumn.floor = __floor

# -----------------------------------------------------------------------------

def __gamma(self):
    """Compute gamma function."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="tgamma",
        operand1=self,
        operand2=None
    )

Column.gamma = __gamma
_VirtualColumn.gamma = __gamma

# -----------------------------------------------------------------------------

def __ge(self, other):
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="greater_equal",
        operand1=self,
        operand2=other
    )

Column.__ge__ = __ge
_VirtualColumn.__ge__ = __ge

# -----------------------------------------------------------------------------

def __get(self, sock=None):
    """
    Transform column to numpy array

    Args:
        sock: Socket connecting the Python API with the getML
            engine.
    """

    # -------------------------------------------
    # Build command string

    cmd = dict()

    cmd["name_"] = self.thisptr["df_name_"]
    cmd["type_"] = "Column.get"

    cmd["col_"] = self.thisptr

    # -------------------------------------------
    # Establish communication with getml engine

    if sock is None:
        s = comm.send_and_receive_socket(cmd)
    else:
        s = sock
        comm.send_string(s, json.dumps(cmd))

    msg = comm.recv_string(s)

    # -------------------------------------------
    # Make sure everything went well, receive data
    # and close connection

    if msg != "Found!":
        s.close()
        raise Exception(msg)

    mat = comm.recv_matrix(s)

    # -------------------------------------------
    # Close connection.

    if sock is None:
        s.close()

    # -------------------------------------------
    # If this is a time stamp, then transform to
    # pd.Timestamp.

    if self.thisptr["type_"] == "Column":
        if self.thisptr["role_"] == "time_stamp" or "time stamp" in self.thisptr["unit_"]:
            shape = mat.shape
            mat = [pd.Timestamp(ts_input=ts, unit="D") for ts in mat.ravel()]
            mat = np.asarray(mat)
            mat.reshape(shape[0], shape[1])

    # -------------------------------------------

    return mat.ravel()

    # -------------------------------------------

Column.get = __get
_VirtualColumn.get = __get

# -----------------------------------------------------------------------------

def __get_categorical(self, sock=None):
    """
    Transform column to numpy array

    Args:
        sock: Socket connecting the Python API with the getML
            engine.
    """

    # -------------------------------------------
    # Build command string

    cmd = dict()

    cmd["name_"] = self.thisptr["df_name_"]
    cmd["type_"] = "CategoricalColumn.get"

    cmd["col_"] = self.thisptr

    # -------------------------------------------
    # Send command to engine
    if sock is None:
        s = comm.send_and_receive_socket(cmd)
    else:
        s = sock
        comm.send_string(s, json.dumps(cmd))

    msg = comm.recv_string(s)

    # -------------------------------------------
    # Make sure everything went well, receive data
    # and close connection

    if msg != "Found!":
        s.close()
        raise Exception(msg)

    mat = comm.recv_categorical_matrix(s)

    # -------------------------------------------
    # Close connection.
    if sock is None:
        s.close()

    # -------------------------------------------

    return mat.ravel()

CategoricalColumn.get = __get_categorical
_VirtualCategoricalColumn.get = __get_categorical

# -----------------------------------------------------------------------------

def __gt(self, other):
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="greater",
        operand1=self,
        operand2=other
    )

Column.__gt__ = __gt
_VirtualColumn.__gt__ = __gt

# -----------------------------------------------------------------------------

def __hour(self):
    """
    Extract hour (of the day) from a time stamp.

    If the column is discrete or numerical, that number will be interpreted
    as the number of days since epoch time (January 1, 1970).
    """
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="hour",
        operand1=self,
        operand2=None
    )

Column.hour = __hour
_VirtualColumn.hour = __hour

# -----------------------------------------------------------------------------

def __is_inf(self):
    """Determine whether the value is infinite."""
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="is_inf",
        operand1=self,
        operand2=None
    )

Column.is_inf = __is_inf
_VirtualColumn.is_inf = __is_inf

# -----------------------------------------------------------------------------

def __is_nan(self):
    """Determine whether the value is nan."""
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="is_nan",
        operand1=self,
        operand2=None
    )

Column.is_nan = __is_nan
_VirtualColumn.is_nan = __is_nan

# -----------------------------------------------------------------------------

def __le(self, other):
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="less_equal",
        operand1=self,
        operand2=other
    )

Column.__le__ = __le
_VirtualColumn.__le__ = __le

# -----------------------------------------------------------------------------

def __lgamma(self):
    """Compute log-gamma function."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="gamma",
        operand1=self,
        operand2=None
    )

Column.lgamma = __lgamma
_VirtualColumn.lgamma = __lgamma

# -----------------------------------------------------------------------------

def __log(self):
    """Compute natural logarithm."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="log",
        operand1=self,
        operand2=None
    )

Column.log = __log
_VirtualColumn.log = __log

# -----------------------------------------------------------------------------

def __lt(self, other):
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="less",
        operand1=self,
        operand2=other
    )

Column.__lt__ = __lt
_VirtualColumn.__lt__ = __lt

# -----------------------------------------------------------------------------

def __max(self, alias="new_column"):
    """
    MAX aggregation.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "max")

Column.max = __max
_VirtualColumn.max = __max

# -----------------------------------------------------------------------------

def __median(self, alias="new_column"):
    """
    MEDIAN aggregation.

    **alias**: Name for the new column.
    """
    return _Aggregation(alias, self, "median")

Column.median = __median
_VirtualColumn.median = __median

# -----------------------------------------------------------------------------

def __min(self, alias="new_column"):
    """
    MIN aggregation.

    **alias**: Name for the new column.
    """
    return _Aggregation(alias, self, "min")

Column.min = __min
_VirtualColumn.min = __min

# -----------------------------------------------------------------------------

def __minute(self):
    """
    Extract minute (of the hour) from a time stamp.

    If the column is discrete or numerical, that number will be interpreted
    as the number of days since epoch time (January 1, 1970).
    """
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="minute",
        operand1=self,
        operand2=None
    )

Column.minute = __minute
_VirtualColumn.minute = __minute

# -----------------------------------------------------------------------------

def __mod(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="fmod",
        operand1=self,
        operand2=other
    )

def __rmod(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="fmod",
        operand1=other,
        operand2=self
    )

Column.__mod__ = __mod
Column.__rmod__ = __rmod

_VirtualColumn.__mod__ = __mod
_VirtualColumn.__rmod__ = __rmod

# -----------------------------------------------------------------------------

def __month(self):
    """
    Extract month from a time stamp.

    If the column is discrete or numerical, that number will be interpreted
    as the number of days since epoch time (January 1, 1970).
    """
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="month",
        operand1=self,
        operand2=None
    )

Column.month = __month
_VirtualColumn.month = __month

# -----------------------------------------------------------------------------

def __mul(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="multiplies",
        operand1=self,
        operand2=other
    )

Column.__mul__ = __mul
Column.__rmul__ = __mul

_VirtualColumn.__mul__ = __mul
_VirtualColumn.__rmul__ = __mul


# -----------------------------------------------------------------------------

def __ne(self, other):
    return _VirtualBooleanColumn(
        df_name=self.thisptr["df_name_"],
        operator="not_equal_to",
        operand1=self,
        operand2=other
    )

Column.__ne__ = __ne
_VirtualColumn.__ne__ = __ne

CategoricalColumn.__ne__ = __ne
_VirtualCategoricalColumn.__ne__ = __ne

# -----------------------------------------------------------------------------

def __neg(self):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="multiplies",
        operand1=self,
        operand2=-1.0
    )

Column.__neg__ = __neg
_VirtualColumn.__neg__ = __neg

# -----------------------------------------------------------------------------

def __pow(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="pow",
        operand1=self,
        operand2=other
    )

def __rpow(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="pow",
        operand1=other,
        operand2=self
    )

Column.__pow__ = __pow
Column.__rpow__ = __rpow

_VirtualColumn.__pow__ = __pow
_VirtualColumn.__rpow__ = __rpow

# -----------------------------------------------------------------------------

def __round(self):
    """Round to nearest."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="round",
        operand1=self,
        operand2=None
    )

Column.round = __round
_VirtualColumn.round = __round

# -----------------------------------------------------------------------------

def __second(self):
    """
    Extract second (of the minute) from a time stamp.

    If the column is discrete or numerical, that number will be interpreted
    as the number of days since epoch time (January 1, 1970).
    """
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="second",
        operand1=self,
        operand2=None
    )

Column.second = __second
_VirtualColumn.second = __second

# -----------------------------------------------------------------------------

def __sin(self):
    """Compute sine."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="sin",
        operand1=self,
        operand2=None
    )

Column.sin = __sin
_VirtualColumn.sin = __sin

# -----------------------------------------------------------------------------

def __sqrt(self):
    """Compute square root."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="sqrt",
        operand1=self,
        operand2=None
    )

Column.sqrt = __sqrt
_VirtualColumn.sqrt = __sqrt

# -----------------------------------------------------------------------------

def __stddev(self, alias="new_column"):
    """
    STDDEV aggregation.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "stddev")

Column.stddev = __stddev
_VirtualColumn.stddev = __stddev

# -----------------------------------------------------------------------------

def __sub(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="minus",
        operand1=self,
        operand2=other
    )

def __rsub(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="minus",
        operand1=other,
        operand2=self
    )

Column.__sub__ = __sub
Column.__rsub__ = __rsub

_VirtualColumn.__sub__ = __sub
_VirtualColumn.__rsub__ = __rsub


# -----------------------------------------------------------------------------

def __substr(self, begin, length):
    """
    Return a substring for every element in the column.

    Args:
        begin (int): First position of the original string.
        length (int): Length of the extracted string.
    """
    col = _VirtualCategoricalColumn(
        df_name=self.thisptr["df_name_"],
        operator="substr",
        operand1=self,
        operand2=None
    )
    col.thisptr["begin_"] = begin
    col.thisptr["len_"] = length
    return col

CategoricalColumn.substr = __substr
_VirtualCategoricalColumn.substr = __substr

# -----------------------------------------------------------------------------

def __sum(self, alias="new_column"):
    """
    SUM aggregation.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "sum")

Column.sum = __sum
_VirtualColumn.sum = __sum

# -----------------------------------------------------------------------------

def __tan(self):
    """Compute tangent."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="tan",
        operand1=self,
        operand2=None
    )

Column.tan = __tan
_VirtualColumn.tan = __tan

# -----------------------------------------------------------------------------

def __to_num(self):
    """Transforms a categorical column to a numeric column."""
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="to_num",
        operand1=self,
        operand2=None
    )

CategoricalColumn.to_num = __to_num
_VirtualCategoricalColumn.to_num = __to_num

# -----------------------------------------------------------------------------

def __to_str(self):
    """Transforms column to a string."""
    return _VirtualCategoricalColumn(
        df_name=self.thisptr["df_name_"],
        operator="to_str",
        operand1=self,
        operand2=None
    )

Column.to_str = __to_str
_VirtualColumn.to_str = __to_str

_VirtualBooleanColumn.to_str = __to_str

# -----------------------------------------------------------------------------

def __to_ts(self, time_formats=["%Y-%m-%dT%H:%M:%s%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]):
    """
    Transforms a categorical column to a time stamp.

    Args:
        time_formats (str): Formats to be used to parse the time stamps.
    """
    col = _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="to_ts",
        operand1=self,
        operand2=None
    )
    col.thisptr["time_formats_"] = time_formats
    return col

CategoricalColumn.to_ts = __to_ts
_VirtualCategoricalColumn.to_ts = __to_ts


# -----------------------------------------------------------------------------

def __truediv(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="divides",
        operand1=self,
        operand2=other
    )

def __rtruediv(self, other):
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="divides",
        operand1=other,
        operand2=self
    )

Column.__truediv__ = __truediv
Column.__rtruediv__ = __rtruediv

_VirtualColumn.__truediv__ = __truediv
_VirtualColumn.__rtruediv__ = __rtruediv

# -----------------------------------------------------------------------------

def __update(self, condition, values):
    """
    Returns an updated version of this column.

    All entries for which the corresponding **condition** is True,
    are updated using the corresponding entry in **values**.

    Args:
        condition (Boolean column): Condition according to which the update is done
        values: Values to update with
    """
    col = _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="update",
        operand1=self,
        operand2=values
    )
    if condition.thisptr["type_"] != "VirtualBooleanColumn":
        raise Exception("Condition for an update must be a Boolen column.")
    col.thisptr["condition_"] = condition.thisptr
    return col

Column.update = __update

_VirtualColumn.update = __update

# -----------------------------------------------------------------------------

def __update_categorical(self, condition, values):
    """
    Returns an updated version of this column.

    All entries for which the corresponding **condition** is True,
    are updated using the corresponding entry in **values**.

    Args:
        condition (Boolean column): Condition according to which the update is done
        values: Values to update with
    """
    col = _VirtualCategoricalColumn(
        df_name=self.thisptr["df_name_"],
        operator="update",
        operand1=self,
        operand2=values
    )
    if condition.thisptr["type_"] != "VirtualBooleanColumn":
        raise Exception("Condition for an update must be a Boolen column.")
    col.thisptr["condition_"] = condition.thisptr
    return col

CategoricalColumn.update = __update_categorical

_VirtualCategoricalColumn.update = __update_categorical

# -----------------------------------------------------------------------------

def __var(self, alias="new_column"):
    """
    VAR aggregation.

    Args:
        alias (str): Name for the new column.
    """
    return _Aggregation(alias, self, "var")

Column.var = __var
_VirtualColumn.var = __var

# -----------------------------------------------------------------------------

def __weekday(self):
    """
    Extract day of the week from a time stamp, Sunday being 0.

    If the column is discrete or numerical, that number will be interpreted
    as the number of days since epoch time (January 1, 1970).
    """
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="weekday",
        operand1=self,
        operand2=None
    )

Column.weekday = __weekday
_VirtualColumn.weekday = __weekday

# -----------------------------------------------------------------------------

def __year(self):
    """
    Extract year from a time stamp.

    If the column is discrete or numerical, that number will be interpreted
    as the number of days since epoch time (January 1, 1970).
    """
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="year",
        operand1=self,
        operand2=None
    )

Column.year = __year
_VirtualColumn.year = __year

# -----------------------------------------------------------------------------

def __yearday(self):
    """
    Extract day of the year from a time stamp.

    If the column is discrete or numerical, that number will be interpreted
    as the number of days since epoch time (January 1, 1970).
    """
    return _VirtualColumn(
        df_name=self.thisptr["df_name_"],
        operator="yearday",
        operand1=self,
        operand2=None
    )

Column.yearday = __yearday
_VirtualColumn.yearday = __yearday


# -----------------------------------------------------------------------------
