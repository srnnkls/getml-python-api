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

import datetime
import json
import os
import platform
import socket
import time
import getml

import getml.communication as comm

from .data_frame import DataFrame

# -----------------------------------------------------------------------------


def delete_project(
        name
):
    """
    Deletes the project.

    All data and models contained in the project directory will be lost.

    Args:
        name (str): Name of your project.

    Raises:
        ConnectionRefusedError: If unable to connect to engine
    """

    cmd = dict()
    cmd["type_"] = "delete_project"
    cmd["name_"] = name

    comm.send(cmd)

# -----------------------------------------------------------------------------

def is_alive():
    """
    Checks if engine is running.

    Returns:
        bool: True if the engine is running and accepting commands, False otherwise 
    """

    ## ---------------------------------------------------------------
    
    cmd = dict()
    cmd["type_"] = "is_alive"
    cmd["name_"] = ""

    s = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )
    try:
        s.connect((getml.host, getml.port))
    except ConnectionRefusedError:
        return False

    comm.send_string(s, json.dumps(cmd))

    s.close()

    return True


# -----------------------------------------------------------------------------

def list_data_frames():
    """
    List all data frames currently stored in the project folder 
    and held in memory.

    Returns:
        dict: Lists the names of the data frames. 
    """

    cmd = dict()
    cmd["type_"] = "list_data_frames"
    cmd["name_"] = ""
    
    s = comm.send_and_receive_socket(cmd)

    msg = comm.recv_string(s)

    if msg != "Success!":
        raise Exception(msg)
    
    json_str = comm.recv_string(s) 
    
    s.close() 

    return json.loads(json_str)

# -----------------------------------------------------------------------------

def list_models():
    """
    List all models currently held in memory.

    Returns:
        dict: Lists the names of all of the models by type. 
    """

    cmd = dict()
    cmd["type_"] = "list_models"
    cmd["name_"] = ""
    
    s = comm.send_and_receive_socket(cmd)

    msg = comm.recv_string(s)

    if msg != "Success!":
        raise Exception(msg)
    
    json_str = comm.recv_string(s) 
    
    s.close() 

    return json.loads(json_str)

# -----------------------------------------------------------------------------

def list_projects():
    """
    List all projects on the engine.

    Returns:
        list: Lists the name all of the projects. 
    """

    cmd = dict()
    cmd["type_"] = "list_projects"
    cmd["name_"] = ""
    
    s = comm.send_and_receive_socket(cmd)

    msg = comm.recv_string(s)

    if msg != "Success!":
        raise Exception(msg)
    
    json_str = comm.recv_string(s) 
    
    s.close() 

    return json.loads(json_str)["projects"]

# -----------------------------------------------------------------------------


def load_data_frame(name):
    """
    Loads a DataFrame object into memory.

    Args:
        name (str): Name of the DataFrame object
    """

    return DataFrame(name).load()

# -----------------------------------------------------------------------------


def run(path):
    """
    Starts the engine

    The engine is started as a background process. Effectively the same
    as calling `./run` or `run.bat` in a separate terminal. The output of the engine
    will be stored in a script called 'log-CURRRENT_DATE.txt'.
    
    Args:
        path (str): Path of the getml engine (where the script run or run.bat can be found).
    """

    cwd = os.getcwd()

    os.chdir(path)

    if platform.system == "Windows":
        os.popen("run.bat >> log-" +
                 datetime.datetime.now().isoformat().split(".")[0].replace(':', '-') + ".txt")
    else:
        os.popen("./run >> log-" +
                 datetime.datetime.now().isoformat().split(".")[0].replace(':', '-') + ".txt")

    while is_alive() == False:
        time.sleep(0.1)

    os.chdir(cwd)

# -----------------------------------------------------------------------------


def setup(path):
    """
    Runs the setup script

    This is only relevant on Mac and Linux. 
    
    Args:
        path (str): Path of the getml engine (where the script setup.sh can be found).
    """
    if platform.system != "Windows":
        cwd = os.getcwd()

        os.chdir(path)

        os.system("sh setup.sh")

        os.chdir(cwd)

# -----------------------------------------------------------------------------


def set_project(
        name
):
    """
    Select a project.

    All data frames and models will be stored in the corresponding project
    directory. If a project of that name does not already exist, a new one will
    be created.

    Args:
        name (str): Name of your project.

    Raises:
        ConnectionRefusedError: If unable to connect to engine
    """
    if not is_alive():
        err_msg = "Cannot connect to getML engine. Make sure the engine is running and you are logged in."
        raise ConnectionRefusedError(err_msg)

    cmd = dict()
    cmd["type_"] = "set_project"
    cmd["name_"] = name

    comm.send(cmd)

# -----------------------------------------------------------------------------


def shutdown():
    """
    Shuts the engine down.

    Raises:
        ConnectionRefusedError: If unable to connect to engine
    """

    cmd = dict()
    cmd["type_"] = "shutdown"
    cmd["name_"] = "all"

    ## In case of the shutdown there will be no returned message to
    ## check the success.
    s = comm.send_and_receive_socket(cmd)

    s.close()

# -----------------------------------------------------------------------------
