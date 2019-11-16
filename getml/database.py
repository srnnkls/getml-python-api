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
This module contains communication routines for the database.
"""

import json
import os

import getml.communication as comm

# -----------------------------------------------------------------------------

def connect_postgres(
    pg_host,
    pg_hostaddr,
    pg_port,
    dbname,
    user,
    password,
    time_formats=["%Y-%m-%dT%H:%M:%s%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]):
    """
    Creates a new PostgreSQL database connection.

    Args:
        pg_host (str): Host of the PostgreSQL database.
        pg_hostaddr (str): IP address of the PostgreSQL database.
        pg_port(int): Port of the PostgreSQL database.
        user (str): User name with which to log into the PostgreSQL database.
        password (str): Password with which to log into the PostgreSQL database.
        time_formats (str, optional): The formats tried when parsing time stamps.
            Check out https://pocoproject.org/docs/Poco.DateTimeFormatter.html#9946 for the options.
    """

    # -------------------------------------------
    # Prepare command.

    cmd = dict()

    cmd["name_"] = ""
    cmd["type_"] = "Database.new"
    cmd["db_"] = "postgres"

    cmd["host_"] = pg_host
    cmd["hostaddr_"] = pg_hostaddr
    cmd["port_"] = pg_port
    cmd["dbname_"] = dbname
    cmd["user_"] = user
    cmd["password_"] = password
    cmd["time_formats_"] = time_formats

    # -------------------------------------------
    # Send JSON command to engine.

    comm.send(cmd)

# -----------------------------------------------------------------------------

def connect_sqlite3(
    name=":memory:",
    time_formats=["%Y-%m-%dT%H:%M:%s%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]):
    """
    Creates a new SQLite3 database connection.

    Args:
        name (str): Name of the sqlite3 file.  If the file does not exist, it
            will be created. Set to ":memory:" for a purely in-memory SQLite3
            database.
        time_formats (str, optional): The formats tried when parsing time stamps.
            Check out https://pocoproject.org/docs/Poco.DateTimeFormatter.html#9946 for the options.
    """

    # -------------------------------------------
    # Prepare command.

    cmd = dict()

    cmd["name_"] = os.path.abspath(name)
    cmd["type_"] = "Database.new"

    cmd["db_"] = "sqlite3"
    cmd["time_formats_"] = time_formats

    # -------------------------------------------
    # Send JSON command to engine.

    comm.send(cmd)

# -----------------------------------------------------------------------------


def drop_table(
    name):
    """
    Drops a table from the database.

    Args:
        name (str): The table to be dropped.
    """

    # -------------------------------------------
    # Prepare command.

    cmd = dict()

    cmd["name_"] = name
    cmd["type_"] = "Database.drop_table"

    # -------------------------------------------
    # Send JSON command to engine.

    comm.send(cmd)

# -----------------------------------------------------------------------------

def execute(
    query):
    """
    Executes an SQL query on the database.

    Args:
        query (str): The SQL query to be executed.
    """

    # -------------------------------------------
    # Prepare command.

    cmd = dict()

    cmd["name_"] = ""
    cmd["type_"] = "Database.execute"

    # -------------------------------------------
    # Send JSON command to engine.

    s = comm.send_and_receive_socket(cmd)

    # -------------------------------------------
    # Send the actual query.

    comm.send_string(s, query)

    # -------------------------------------------
    # Make sure that everything went well.

    msg = comm.recv_string(s)

    s.close()

    if msg != "Success!":
        raise Exception(msg)

# -----------------------------------------------------------------------------

def get_colnames(
    name):
    """
    Lists the colnames of a table held in the database.

    Args:
        name (str): The name of the database.
    """
    # -------------------------------------------
    # Prepare command.

    cmd = dict()

    cmd["name_"] = name
    cmd["type_"] = "Database.get_colnames"

    # -------------------------------------------
    # Send JSON command to engine.

    s = comm.send_and_receive_socket(cmd)

    # -------------------------------------------
    # Make sure that everything went well.

    msg = comm.recv_string(s)

    if msg != "Success!":
        s.close()
        raise Exception(msg)

    # -------------------------------------------
    # Parse result as list.

    arr = json.loads(comm.recv_string(s))

    s.close()

    return arr


# -----------------------------------------------------------------------------


def list_tables():
    """
    Lists all tables and views currently held in the database.
    """
    
    # -------------------------------------------
    # Prepare command.

    cmd = dict()

    cmd["name_"] = ""
    cmd["type_"] = "Database.list_tables"

    # -------------------------------------------
    # Send JSON command to engine.

    s = comm.send_and_receive_string(cmd)

    # -------------------------------------------
    # Make sure that everything went well.

    msg = comm.recv_string(s)

    if msg != "Success!":
        s.close()
        raise Exception(msg)

    # -------------------------------------------
    # Parse result as list.

    arr = json.loads(comm.recv_string(s))

    s.close()

    return arr

# -----------------------------------------------------------------------------


def read_csv(
    name,
    fnames,
    header=True,
    quotechar='"',
    sep=',',
    skip=0):
    """
    Reads a CSV file into the database.

    Args:
        name (str): Name of the table in which the data is to be inserted.
        fnames (List[str]): The list of CSV file names to be read.
        header (bool, optional): Whether the CSV file contains a header with the column names. Default to True.
        quotechar (str, optional): The character used to wrap strings. Default:`"`
        sep (str, optional): The separator used for separating fields. Default:`,`
        skip (int, optional): Number of lines to skip at the beginning of each
            file (Default: 0). If *header* is True, the lines will be skipped
            before the header.
    """
    # -------------------------------------------
    # Transform paths
    fnames_ = [os.path.abspath(_) for _ in fnames]
    
    # -------------------------------------------
    # Prepare command.

    cmd = dict()

    cmd["name_"] = name
    cmd["type_"] = "Database.read_csv"

    cmd["fnames_"] = fnames_
    cmd["header_"] = header
    cmd["quotechar_"] = quotechar
    cmd["sep_"] = sep
    cmd["skip_"] = skip

    # -------------------------------------------
    # Send JSON command to engine.

    comm.send(cmd)

# -----------------------------------------------------------------------------


def sniff_csv(
    name,
    fnames,
    header=True,
    num_lines_sniffed=1000,
    quotechar='"',
    sep=',',
    skip=0):
    """
    Sniffs a list of CSV files.

    Args:
        name (str): Name of the table in which the data is to be inserted.
        fnames (List[str]): The list of CSV file names to be read.
        header (bool, optional): Whether the CSV file contains a header with the column names. Default to True.
        quotechar (str, optional): The character used to wrap strings. Default:`"`
        sep (str, optional): The separator used for separating fields. Default:`,`
        skip (int, optional): Number of lines to skip at the beginning of each
            file (Default: 0). If *header* is True, the lines will be skipped
            before the header.

    Returns:
        str: Appropriate `CREATE TABLE` statement.
    """
    # -------------------------------------------
    # Transform paths
    fnames_ = [os.path.abspath(_) for _ in fnames]

    # -------------------------------------------
    # Prepare command.

    cmd = dict()

    cmd["name_"] = name
    cmd["type_"] = "Database.sniff_csv"

    cmd["fnames_"] = fnames_
    cmd["header_"] = header
    cmd["num_lines_sniffed_"] = num_lines_sniffed
    cmd["quotechar_"] = quotechar
    cmd["sep_"] = sep
    cmd["skip_"] = skip

    # -------------------------------------------
    # Send JSON command to engine.

    s = comm.send_and_receive_socket(cmd)

    # -------------------------------------------
    # Make sure that everything went well.

    msg = comm.recv_string(s)

    if msg != "Success!":
        s.close()
        raise Exception(msg)

    # -------------------------------------------

    query = comm.recv_string(s)

    s.close()

    return query

# -----------------------------------------------------------------------------
