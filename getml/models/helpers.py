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

import getml.communication as comm

from .multirel_model import MultirelModel
from .relboost_model import RelboostModel

# -----------------------------------------------------------------------------

def get_model(
        name
):
    """
    Returns a handle to the model specified by name.

    Args:
        name (str): Name of the model.

    """

    cmd = dict()
    cmd["type_"] = "get_model"
    cmd["name_"] = name

    s = comm.send_and_receive_socket(cmd)

    msg = comm.recv_string(s) 

    if msg == "MultirelModel":
        return MultirelModel(name=name).refresh()
    elif msg == "RelboostModel":
        return RelboostModel(name=name).refresh()
    else:
        raise Exception(msg)


# -----------------------------------------------------------------------------
