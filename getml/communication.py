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
Handles the communication for the getml library
"""

import json
import socket
import sys
import warnings

import numpy as np

# ----------------------------------------------------

def recv_data(socket, size):
    """
    Receives data (of any type) sent by the getml engine
    """

    data = []

    bytes_received = np.uint64(0)

    max_chunk_size = np.uint64(2048)

    while bytes_received < size:

        current_chunk_size = int(min(size - bytes_received, max_chunk_size))

        chunk = socket.recv(current_chunk_size)

        if chunk == '':
            raise RuntimeError("Connection to getml engine broken")

        data.append(chunk)

        bytes_received += np.uint64(len(chunk))

    return ''.encode().join(data)


# ----------------------------------------------------

def recv_boolean_matrix(socket):
    """
    Receives a matrix (type boolean) from the getml engine.
    """

    # By default, numeric data sent over the socket is big endian,
    # also referred to as network-byte-order!
    if sys.byteorder == 'little':

        shape = np.frombuffer(
            socket.recv(np.nbytes[np.int32] * 2),
            dtype=np.int32
        ).byteswap().astype(np.uint64)

        size = shape[0] * shape[1] * np.uint64(np.nbytes[np.int32])

        matrix = recv_data(socket, size)

        matrix = np.frombuffer(
            matrix,
            dtype=np.int32
        ).byteswap()

        matrix = matrix.reshape(shape[0], shape[1])

    else:

        shape = np.frombuffer(
            socket.recv(np.nbytes[np.int32] * 2),
            dtype=np.int32
        ).astype(np.uint64)

        size = shape[0] * shape[1] * np.uint64(np.nbytes[np.int32])

        matrix = recv_data(socket, size)

        matrix = np.frombuffer(
            matrix,
            dtype=np.int32
        )

        matrix = matrix.reshape(shape[0], shape[1])

    return (matrix == 1)

# ----------------------------------------------------


def recv_matrix(socket):
    """
    Receives a matrix (type np.float64) from the getml engine.
    """

    # By default, numeric data sent over the socket is big endian,
    # also referred to as network-byte-order!
    if sys.byteorder == 'little':

        shape = np.frombuffer(
            socket.recv(np.nbytes[np.int32] * 2),
            dtype=np.int32
        ).byteswap().astype(np.uint64)

        size = shape[0] * shape[1] * np.uint64(np.nbytes[np.float64])

        matrix = recv_data(socket, size)

        matrix = np.frombuffer(
            matrix,
            dtype=np.float64
        ).byteswap()

        matrix = matrix.reshape(shape[0], shape[1])

    else:

        shape = np.frombuffer(
            socket.recv(np.nbytes[np.int32] * 2),
            dtype=np.int32
        ).astype(np.uint64)

        size = shape[0] * shape[1] * np.uint64(np.nbytes[np.float64])

        matrix = recv_data(socket, size)

        matrix = np.frombuffer(
            matrix,
            dtype=np.float64
        )

        matrix = matrix.reshape(shape[0], shape[1])

    return matrix

# ----------------------------------------------------


def recv_string(socket):
    """
    Receives a string from the getml engine
    (an actual string, not a bytestring).
    """

    # By default, numeric data sent over the socket is big endian,
    # also referred to as network-byte-order!
    if sys.byteorder == 'little':

        size = np.frombuffer(
            socket.recv(np.nbytes[np.int32]),
            dtype=np.int32
        ).byteswap()[0]

    else:

        size = np.frombuffer(
            socket.recv(np.nbytes[np.int32]),
            dtype=np.int32
        )[0]

    string = recv_data(socket, size).decode()

    return string


# ----------------------------------------------------

# Depends on recv_string, so it is not in alphabetical order.
def recv_categorical_matrix(socket):
    """
    Receives a matrix of type string from the getml engine
    """

    # -------------------------------------------------------------------------
    # Receive shape

    # By default, numeric data sent over the socket is big endian,
    # also referred to as network-byte-order!
    if sys.byteorder == 'little':

        shape = np.frombuffer(
            socket.recv(np.nbytes[np.int32] * 2),
            dtype=np.int32
        ).byteswap().astype(np.uint64)

        size = shape[0] * shape[1]

    else:

        shape = np.frombuffer(
            socket.recv(np.nbytes[np.int32] * 2),
            dtype=np.int32
        ).astype(np.uint64)

        size = shape[0] * shape[1]

    # -------------------------------------------------------------------------
    # Receive actual strings

    mat = []

    for i in range(size):
        mat.append(recv_string(socket))

    # -------------------------------------------------------------------------
    # Cast as numpy.array and reshape

    mat = np.asarray(mat)

    return mat.reshape(shape[0], shape[1])

# ----------------------------------------------------


def send_string(socket, string):
    """
    Sends a string to the getml engine
    (an actual string, not a bytestring).
    """

    encoded = string.encode('utf-8')

    size = len(encoded)

    # By default, numeric data sent over the socket is big endian,
    # also referred to as network-byte-order!
    if sys.byteorder == 'little':

        socket.sendall(
            np.asarray([size]).astype(np.int32).byteswap().tostring()
        )

    else:

        socket.sendall(
            np.asarray([size]).astype(np.int32).tostring()
        )

    socket.sendall(encoded)


# ----------------------------------------------------

def send_categorical_matrix(socket, categorical_matrix):
    """
    Sends a list of strings to the getml engine
    (an actual string, not a bytestring).
    """

    if len(categorical_matrix.shape) == 1:
        shape = [categorical_matrix.shape[0], 1]

    elif len(categorical_matrix.shape) > 2:
        raise Exception(
            "Numpy array must be one-dimensional or two-dimensional!"
        )

    else:
        shape = categorical_matrix.shape

    # By default, numeric data sent over the socket is big endian,
    # also referred to as network-byte-order!
    if sys.byteorder == 'little':
        # Sends the shape of the categorical matrix.
        socket.sendall(
            np.asarray(shape).astype(np.int32).byteswap().tostring()
        )

        # Sends the length of the encoded string, followed by the encoded string
        # itself.
        socket.sendall(
            ''.encode().join([
                np.asarray(len(elem.encode('utf-8'))
                           ).astype(np.int32).byteswap().tostring() + elem.encode('utf-8')
                for elem in categorical_matrix.astype(str).flatten()
            ])
        )

    else:
        # Sends the shape of the categorical matrix.
        socket.sendall(
            np.asarray(shape).astype(np.int32).tostring()
        )

        # Sends the length of the encoded string, followed by the encoded string
        # itself.
        socket.sendall(
            ''.encode().join([
                np.asarray(len(elem.encode('utf-8'))
                           ).astype(np.int32).tostring() + elem.encode('utf-8')
                for elem in categorical_matrix.astype(str).flatten()
            ])
        )

# ----------------------------------------------------


def send_cmd(socket, cmd_string):
    """
    Sends a command to the getml engine.
    """

    send_string(socket, cmd_string)

# ----------------------------------------------------


def send_matrix(socket, matrix):
    """
    Sends a matrix (type np.float64) to the getml engine.
    """

    if len(matrix.shape) == 1:
        shape = [matrix.shape[0], 1]

    elif len(matrix.shape) > 2:
        raise Exception(
            "Numpy array must be one-dimensional or two-dimensional!"
        )

    else:
        shape = matrix.shape

    # By default, numeric data sent over the socket is big endian,
    # also referred to as network-byte-order!
    if sys.byteorder == 'little':

        socket.sendall(
            np.asarray(shape).astype(np.int32).byteswap().tostring()
        )

        socket.sendall(
            matrix.astype(np.float64).byteswap().tostring()
        )

    else:

        socket.sendall(
            np.asarray(shape).astype(np.int32).tostring()
        )

        socket.sendall(
            matrix.astype(np.float64).tostring()
        )

# ----------------------------------------------------
