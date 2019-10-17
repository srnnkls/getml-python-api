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
This module contains the loss functions for the getml library.
"""

# ------------------------------------------------------------------------------


class _LossFunction(object):
    """
    Base class. Should not ever be directly initialized!
    """

    def __init__(self):
        self.thisptr = dict()
        self.thisptr["type_"] = "none"

# ------------------------------------------------------------------------------


class CrossEntropyLoss(_LossFunction):
    """
    Cross entropy function.

    Recommended loss function for classification problems.
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.thisptr["type_"] = "CrossEntropyLoss"


# ------------------------------------------------------------------------------


class SquareLoss(_LossFunction):
    """
    Square loss function.

    Recommended loss function for regression problems.
    """

    def __init__(self):
        super(SquareLoss, self).__init__()
        self.thisptr["type_"] = "SquareLoss"

# ------------------------------------------------------------------------------
