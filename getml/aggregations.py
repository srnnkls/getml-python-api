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
This module contains the aggregations for the getml library.
"""

# ------------------------------------------------------------------------------

Avg = "AVG"
"""Average value of a given numeric column.
"""
Count = "COUNT"
"""Number of rows in a given column.
"""
CountDistinct = "COUNT DISTINCT"
"""Count function with distinct clause.

The distinct clause eliminates the repetitive appearance of the same data.
"""
CountMinusCountDistinct = "COUNT MINUS COUNT DISTINCT"
"""Counts minus counts distinct.
"""
Max = "MAX"
"""Largest value of a given column.
"""
Median = "MEDIAN"
"""Median of a given column
"""
Min = "MIN"
"""Smallest value of a given column.
"""
Stddev = "STDDEV"
"""Standard deviation of a given column.
"""
Sum = "SUM"
"""Total sum of a given numeric column.
"""
Var = "VAR"
"""Statistical variance of a given numeric column.
"""

# ------------------------------------------------------------------------------



