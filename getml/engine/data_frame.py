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

import json
import os

import numpy as np
import pandas as pd

from getml.placeholder import Placeholder

import getml.columns as columns
import getml.communication as comm

# -----------------------------------------------------------------------------


class DataFrame(object):
    """
    Data storage container for the getml engine.

    Args:
        name (str): Name of the DataFrame
        join_keys (List[str], optional) : Names of the columns that are the join keys.
        time_stamps (List[str], optional): Names of the columns that are the
            time stamps.  Time stamps can be of type pd.Timestamp or float. If
            they are a float, the floating point number will be interpreted as
            the number of days since 1970-01-01 00:00:00. Fractions will be
            interpreted as fractions of a day. For instance, 2.5 will be
            interpreted as 1970-01-03 12:00:00.

        categorical (List[str], optional): Names of the columns that are categorical variables.
        discrete (List[str]), optional) : Names of the columns that are discrete variables.
        numerical (List[str]): Names of the columns that are numerical variables.
        targets (List[str], optional): Target variables.
            Will be ignored during prediction or if this is peripheral table.
        units (dict): Mapping of column names to units.
            All columns containing that column name will be assigned the unit.
            Columns containing the same unit can be directly compared.

    """
    
    def __init__(
        self,
        name,
        join_keys=None,
        time_stamps=None,
        categorical=None,
        discrete=None,
        numerical=None,
        targets=None,
        units=None
    ):

        # ---------------------------------------------------------------------

        self.name = name

        self.units = units or dict()

        # ---------------------------------------------------------------------
        
        join_key_names = join_keys or []
        time_stamp_names = time_stamps or []
        categorical_names = categorical or []
        discrete_names = discrete or []
        numerical_names = numerical or []
        target_names = targets or []
        
        # ---------------------------------------------------------------------

        self.__categorical_columns = []

        for i, name in enumerate(categorical_names):
            self.__categorical_columns.append(
                columns.CategoricalColumn(
                    name=name,
                    role="categorical",
                    num=i,
                    df_name=self.name
                )
            )

        # ---------------------------------------------------------------------

        discrete_units = self.__extract_units(discrete_names)

        self.__discrete_columns = []

        for i, name in enumerate(discrete_names):
            self.__discrete_columns.append(
                columns.Column(
                    name=name,
                    unit=discrete_units[i],
                    role="discrete",
                    num=i,
                    df_name=self.name
                )
            )

        # ---------------------------------------------------------------------

        self.__join_key_columns = []

        for i, name in enumerate(join_key_names):
            self.__join_key_columns.append(
                columns.CategoricalColumn(
                    name=name,
                    role="join_key",
                    num=i,
                    df_name=self.name
                )
            )

        # ---------------------------------------------------------------------

        numerical_units = self.__extract_units(numerical_names)

        self.__numerical_columns = []

        for i, name in enumerate(numerical_names):
            self.__numerical_columns.append(
                columns.Column(
                    name=name,
                    unit=numerical_units[i],
                    role="numerical",
                    num=i,
                    df_name=self.name
                )
            )

        # ---------------------------------------------------------------------

        self.__target_columns = []

        for i, name in enumerate(target_names):
            self.__target_columns.append(
                columns.Column(
                    name=name,
                    role="target",
                    num=i,
                    df_name=self.name
                )
            )

        # ---------------------------------------------------------------------

        self.__time_stamp_columns = []

        for i, name in enumerate(time_stamp_names):
            self.__time_stamp_columns.append(
                columns.Column(
                    name=name,
                    role="time_stamp",
                    num=i,
                    df_name=self.name
                )
            )

    # -------------------------------------------------------------------------

    def __add_categorical_column(self, col, name, role, unit):

        # ------------------------------------------------------
        # Send command

        cmd = dict()
        cmd["type_"] = "DataFrame.add_categorical_column"
        cmd["name_"] = name

        cmd["col_"] = col.thisptr
        cmd["df_name_"] = self.name
        cmd["role_"] = role
        cmd["unit_"] = unit

        comm.send(cmd)

        # ------------------------------------------------------

        self.refresh()

    # -------------------------------------------------------------------------

    def __add_column(self, col, name, role, unit):

        # ------------------------------------------------------
        # Send command

        cmd = dict()
        cmd["type_"] = "DataFrame.add_column"
        cmd["name_"] = name

        cmd["col_"] = col.thisptr
        cmd["df_name_"] = self.name
        cmd["role_"] = role
        cmd["unit_"] = unit

        comm.send(cmd)

        # ------------------------------------------------------

        self.refresh()

    # -------------------------------------------------------------------------

    def __check_plausibility(self, data_frame):

        # ------------------------------------------------------

        if len(self.join_key_names) == 0:
            raise Exception("You need to provide at least one join key!")

        if len(self.time_stamp_names) == 0:
            raise Exception("You need to provide at least one time stamp!")

        if len(self.categorical_names) != len(np.unique(self.categorical_names)):
            raise Exception("Categorical columns not unique!")

        if len(self.discrete_names) != len(np.unique(self.discrete_names)):
            raise Exception("Discrete columns not unique!")

        if len(self.join_key_names) != len(np.unique(self.join_key_names)):
            raise Exception("Join keys not unique!")

        if len(self.numerical_names) != len(np.unique(self.numerical_names)):
            raise Exception("Numerical columns not unique!")

        if len(self.target_names) != len(np.unique(self.target_names)):
            raise Exception("Target columns not unique!")

        if len(self.time_stamp_names) != len(np.unique(self.time_stamp_names)):
            raise Exception("Time stamps not unique!")

        # ------------------------------------------------------

        for col in self.categorical_names:
            if col not in data_frame.columns:
                raise ValueError(
                    "Column named '" + col + "' does not exist!")

        for col in self.discrete_names:
            if col not in data_frame.columns:
                raise ValueError(
                    "Column named '" + col + "' does not exist!")

        for col in self.join_key_names:
            if col not in data_frame.columns:
                raise ValueError(
                    "Column named '" + col + "' does not exist!")

        for col in self.numerical_names:
            if col not in data_frame.columns:
                raise ValueError(
                    "Column named '" + col + "' does not exist!")

        for col in self.target_names:
            if col not in data_frame.columns:
                raise ValueError(
                    "Column named '" + col + "' does not exist!")

        for col in self.time_stamp_names:
            if col not in data_frame.columns:
                raise ValueError(
                    "Column named '" + col + "' does not exist!")

    # -------------------------------------------------------------------------

    def __close(self, s):

        cmd = dict()
        cmd["type_"] = "DataFrame.close"
        cmd["name_"] = self.name

        comm.send_string(s, json.dumps(cmd))
        
        msg = comm.recv_string(s)
        
        if msg != "Success!":
            raise Exception(msg)
        
    # -------------------------------------------------------------------------

    def __extract_shape(self, cmd, name):
        shape = cmd[name + "_shape_"]
        shape = np.asarray(shape).astype(np.int32)
        return shape.tolist()

    # -------------------------------------------------------------------------

    def __extract_units(self, colnames):
        return [
            self.units[col] if col in self.units else "" for col in colnames
        ]

    # -------------------------------------------------------------------------

    def __get_column(self, name, columns):
        for col in columns:
            if col.name == name:
                return col
        raise Exception("Column named '" + name + "' not found.")

    # -------------------------------------------------------------------------


    def __send_data(self, data_frame, s):
        
        for col in self.__categorical_columns:
            col.send(
                data_frame[[col.name]].values.astype(np.str),
                s
            )

        for col in self.__discrete_columns:
            if "time stamp" in col.thisptr["unit_"]:
                col.send(
                    self.__transform_timestamps(
                        data_frame[[col.name]]),
                    s
                )
            else:
                col.send(
                    data_frame[[col.name]].apply(
                        pd.to_numeric, errors="coerce"
                    ).values,
                    s
                )
                
        for col in self.__join_key_columns:
            col.send(
                data_frame[[col.name]].values.astype(np.str),
                s
            )

        for col in self.__numerical_columns:
            if "time stamp" in col.thisptr["unit_"]:
                col.send(
                    self.__transform_timestamps(
                        data_frame[[col.name]]
                    ),
                    s
                )
            else:
                col.send(
                    data_frame[[col.name]].apply(
                        pd.to_numeric, errors="coerce"
                    ).values,
                    s
                )
                
        for col in self.__target_columns:
            col.send(
                data_frame[[col.name]].apply(
                    pd.to_numeric, errors="raise"
                ).values,
                s
            )
            
        for col in self.__time_stamp_columns:
            col.send(
                self.__transform_timestamps(
                    data_frame[[col.name]]
                ),
                s
            )
            
    # -------------------------------------------------------------------------

    def __rm_col(self, name, role):

        # ------------------------------------------------------
        # Send command

        cmd = dict()
        cmd["type_"] = "DataFrame.remove_column"
        cmd["name_"] = name

        cmd["df_name_"] = self.name
        cmd["role_"] = role

        comm.send(cmd)

        # ------------------------------------------------------

        self.refresh()

    # -------------------------------------------------------------------------

    def __transform_timestamps(self, time_stamps):
        # Transforming a time stamp using to_numeric
        # will result in the number of nanoseconds since
        # the beginning of UNIX time. There are 8.64e+13 nanoseconds
        # in a day.
        transformed = pd.DataFrame()

        for colname in time_stamps.columns:
            if pd.api.types.is_numeric_dtype(time_stamps[colname]):
                transformed[colname] = time_stamps[colname]
            else:
                transformed[colname] = time_stamps[[colname]].apply(
                    pd.to_datetime,
                    errors="coerce"
                ).apply(
                    pd.to_numeric,
                    errors="coerce"
                ).apply(
                    lambda val: val / 8.64e+13
                )[colname]

        return transformed.values

    # -------------------------------------------------------------------------

    def add_categorical(self, col, name, unit=""):
        """
        Adds a categorical column to the DataFrame.

        Args:
            col: The column to be added.
            name (str): Name of the new column in the DataFrame.
            unit (otional): Unit of the column.
        """
        self.__add_categorical_column(col, name, "categorical", unit)

    # -------------------------------------------------------------------------

    def add_discrete(self, col, name, unit=""):
        """
        Adds a discrete column to the DataFrame.

        Args:
            col: The column to be added.
            name (str): Name of the new column in the DataFrame.
            unit (otional): Unit of the column.
        """
        self.__add_column(col, name, "discrete", unit)

    # -------------------------------------------------------------------------

    def add_join_key(self, col, name):
        """
        Adds a join key column to the DataFrame.

        Args:
            col: The column to be added.
            name (str): Name of the new column in the DataFrame.
        """
        self.__add_categorical_column(col, name, "join_key", "")

    # -------------------------------------------------------------------------

    def add_numerical(self, col, name, unit=""):
        """
        Adds a numerical column to the DataFrame.

        Args:
            col: The column to be added.
            name (str): Name of the new column in the DataFrame.
            unit (otional): Unit of the column.
        """
        self.__add_column(col, name, "numerical", unit)

    # -------------------------------------------------------------------------

    def add_target(self, col, name):
        """
        Adds a target column to the DataFrame.

        Args:
            col: The column to be added.
            name (str): Name of the new column in the DataFrame.
        """
        self.__add_column(col, name, "target", "")

    # -------------------------------------------------------------------------

    def add_time_stamp(self, col, name):
        """
        Adds a time stamp column to the DataFrame.

        Args:
            col: The column to be added.
            name (str): Name of the new column in the DataFrame.
        """
        self.__add_column(col, name, "time_stamp", "")

    # -------------------------------------------------------------------------

    def append(self, data_frame, sock=None):
        """Appends data to tables that already exist on the getml engine.

        If sock is None, it will call a function to create a new
        socket, use it for the data transfer and close it
        afterwards. If, instead, a socket is provided, it just sends
        all the data but does not close it.

        Args:
            data_frame (pandas.DataFrame): Table that you want to be
                appended to the existing data.
            sock (optional): Socket connecting the Python API with the
                getML engine.

        """

        # ------------------------------------------------------

        self.__check_plausibility(data_frame)

        # ------------------------------------------------------
        # Create connection.

        cmd = dict()
        cmd["type_"] = "DataFrame.append"
        cmd["name_"] = self.name

        if sock is None:
            s = comm.send_and_receive_socket(cmd)
        else:
            s = sock
            comm.send_string(s, json.dumps(cmd))

        # ------------------------------------------------------
        # Send individual matrices to getml engine

        self.__send_data(data_frame, s)

        # ------------------------------------------------------

        self.__close(s)
        
        if sock is None:
            s.close()

        return self

    # -------------------------------------------------------------------------

    def categorical(self, name):
        """
        Handle to a categorical column.
        
        Args:
            name (str): Name of the column.
        """
        return self.__get_column(name, self.__categorical_columns)

    # -------------------------------------------------------------------------
    
    @property
    def categorical_names(self):
        """
        List of the names of all categorical columns.
        """
        return [col.name for col in self.__categorical_columns]
    
    # -------------------------------------------------------------------------

    def delete(self, mem_only=False):
        """
        Deletes the data frame from the engine.

        Args:
            mem_only (bool): If True, the data frame will be deleted from
                memory only, but not from disk.
        """

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "DataFrame.delete"
        cmd["name_"] = self.name
        cmd["mem_only_"] = mem_only

        comm.send(cmd)

    # -------------------------------------------------------------------------

    def discrete(self, name):
        """
        Returns a handle to a discrete column.

        Args:
            name (str): Name of the column.
        """
        return self.__get_column(name, self.__discrete_columns)

    # -------------------------------------------------------------------------
    
    @property
    def discrete_names(self):
        """
        List of the names of all discrete columns.
        """
        return [col.name for col in self.__discrete_columns]
    
    # -------------------------------------------------------------------------

    def from_db(self, table_name, append=False):
        """
        Fill from Database

        The DataFrame will be filled from a table in the database.
        
        Args:
            table_name(str): Table from which we want to retrieve the data.
            append(bool): If a DataFrame already exists, should table be appended?
        """

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "DataFrame.from_db"
        cmd["name_"] = self.name
        cmd["table_name_"] = table_name

        cmd["categoricals_"] = self.categorical_names
        cmd["discretes_"] = self.discrete_names
        cmd["join_keys_"] = self.join_key_names
        cmd["numericals_"] = self.numerical_names
        cmd["targets_"] = self.target_names
        cmd["time_stamps_"] = self.time_stamp_names

        cmd["append_"] = append
        
        comm.send(cmd)

        # -------------------------------------------

        return self

    # -------------------------------------------------------------------------

    def from_json(self, json_str, append=False, time_formats=["%Y-%m-%dT%H:%M:%s%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]):
        """
        Fill from JSON

        Fills the data frame with data from a JSON string. 

        Args:
            json_str (str): The JSON string containing the data.
            append (bool): If a DataFrame already exists, should json_str be appended?
            time_formats (str): The formats tried when parsing time stamps.
                Refer to https://pocoproject.org/docs/Poco.DateTimeFormatter.html#9946 for the options.
        """


        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "DataFrame.from_json"
        cmd["name_"] = self.name

        cmd["categoricals_"] = self.categorical_names
        cmd["discretes_"] = self.discrete_names
        cmd["join_keys_"] = self.join_key_names
        cmd["numericals_"] = self.numerical_names
        cmd["targets_"] = self.target_names
        cmd["time_stamps_"] = self.time_stamp_names

        cmd["append_"] = append
        cmd["time_formats_"] = time_formats

        s = comm.send_and_receive_socket(cmd)

        # -------------------------------------------
        # Send the JSON string

        comm.send_string(s, json_str)

        # -------------------------------------------
        # Make sure everything went well and close
        # connection

        msg = comm.recv_string(s)

        s.close()

        if msg != "Success!":
            raise Exception(msg)

        # -------------------------------------------

        return self

    # -------------------------------------------------------------------------

    def from_query(self, query, append=False):
        """
        Fill from query

        Fills the data frame with data from a table in the database.

        Args:
            query: The query used to retrieve the data. 
            append (bool): If a DataFrame already exists, should table be appended?
        """

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "DataFrame.from_query"
        cmd["name_"] = self.name
        cmd["query_"] = query

        cmd["categoricals_"] = self.categorical_names
        cmd["discretes_"] = self.discrete_names
        cmd["join_keys_"] = self.join_key_names
        cmd["numericals_"] = self.numerical_names
        cmd["targets_"] = self.target_names
        cmd["time_stamps_"] = self.time_stamp_names

        cmd["append_"] = append
        
        comm.send(cmd)

        # -------------------------------------------

        return self

    # -------------------------------------------------------------------------

    def get(self):
        """
        Get Pandas DataFrame

        Returns:
            :class:`pandas.DataFrame`
        """

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "DataFrame.get"
        cmd["name_"] = self.name

        # -------------------------------------------
        # Establish communication with getml engine

        s = comm.send_and_receive_socket(cmd)

        # -------------------------------------------
        # Receive all columns

        df = pd.DataFrame()

        for col in self.__categorical_columns:
            df[col.name] = col.get(s)

        for col in self.__discrete_columns:
            df[col.name] = col.get(s)

        for col in self.__join_key_columns:
            df[col.name] = col.get(s)

        for col in self.__numerical_columns:
            df[col.name] = col.get(s)

        for col in self.__target_columns:
            df[col.name] = col.get(s)

        for col in self.__time_stamp_columns:
            df[col.name] = col.get(s)

        # -------------------------------------------
        # Close connection

        self.__close(s)

        s.close()

        # -------------------------------------------

        return df

    # -------------------------------------------------------------------------

    def group_by(self, join_key, name, aggregations):
        """
        Creates new DataFrame by grouping over a join key.

        Args:
            join_key (str): Name of the join key to group by.
            name (str): Name of the new DataFrame.
            aggregations: List containing aggregations.

        Returns:    
            :class:`~getml.engine.DataFrame`
        """

        # ----------------------------------------------------------------------
        # Build command

        cmd = dict()
        cmd["name_"] = name
        cmd["type_"] = "DataFrame.group_by"

        cmd["join_key_name_"] = join_key
        cmd["df_name_"] = self.name
        cmd["aggregations_"] = [agg.thisptr for agg in aggregations]

        comm.send(cmd)

        # ----------------------------------------------------------------------
        # Create handle for new data frame.

        new_df = DataFrame(name)

        return new_df.refresh()

    # -------------------------------------------------------------------------

    def join(
        self,
        name,
        other,
        join_key,
        other_join_key=None,
        cols=None,
        other_cols=None,
        how="inner",
        where=None):
        """
        Create a new DataFrame by joining this DataFrame with another DataFrame.

        Args:
            name (str): The name of the new DataFrame.
            other (DataFrame): The other DataFrame.
            join_key (str): Name of the join key in this DataFrame.
            other_join_key (str, optional): Name of the join key in the other table
                (if not identical to join_key).
            cols (optional): List of columns from this DataFrame to be included.
                If left blank, all columns from this DataFrame will be included.
            other_cols (optional): List of columns from the other DataFrame to be included.
                If left blank, all columns from the other DataFrame will be included.
            how (str): Type of the join. Supports "left", "right" and "inner".
            where (bool): Boolean column that imposes WHERE conditions on the join.
        """

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "DataFrame.join"
        cmd["name_"] = name

        cmd["df1_name_"] = self.name
        cmd["df2_name_"] = other.name

        cmd["join_key_used_"] = join_key
        cmd["other_join_key_used_"] = other_join_key or join_key

        cmd["cols1_"] = cols or []
        cmd["cols2_"] = other_cols or []

        cmd["cols1_"] = [c.thisptr for c in cmd["cols1_"]]
        cmd["cols2_"] = [c.thisptr for c in cmd["cols2_"]]

        cmd["how_"] = how

        if where is not None:
            cmd["where_"] = where.thisptr

        comm.send(cmd)

        # -------------------------------------------

        return DataFrame(name=name).refresh()

    # -------------------------------------------------------------------------

    def join_key(self, name):
        """
        Returns a handle to a join key.

        Args:
            name (str): Name of the join key.
        """
        return self.__get_column(name, self.__join_key_columns)

    # -------------------------------------------------------------------------
    
    @property
    def join_key_names(self):
        """
        List of the names of all join keys.
        """
        return [col.name for col in self.__join_key_columns]
    
    # -------------------------------------------------------------------------

    def load(self):
        """
        Loads the DataFrame object from the engine.
        """

        # ----------------------------------------------------------------------

        cmd = dict()
        cmd["type_"] = "DataFrame.load"
        cmd["name_"] = self.name

        comm.send(cmd)

        # ----------------------------------------------------------------------

        return self.refresh()

    # -------------------------------------------------------------------------

    def n_bytes(self):
        """
        Returns the size of the data stored in the DataFrame in bytes.
        """

        # ------------------------------------------------------
        # Build and send JSON command

        cmd = dict()
        cmd["type_"] = "DataFrame.nbytes"
        cmd["name_"] = self.name

        s = comm.send_and_receive_socket(cmd)

        # ------------------------------------------------------
        # Make sure model exists on getml engine

        msg = comm.recv_string(s)

        if msg != "Found!":
            s.close()
            raise Exception(msg)

        # ------------------------------------------------------
        # Receive number of bytes from getml engine

        nbytes = comm.recv_string(s)

        # ------------------------------------------------------

        s.close()

        return np.uint64(nbytes)

    # -------------------------------------------------------------------------
    
    @property
    def n_categorical(self):
        """
        Number of categorical columns.
        """
        return len(self.__categorical_columns)
    
    # -------------------------------------------------------------------------
    
    @property
    def n_discrete(self):
        """
        Number of discrete columns.
        """
        return len(self.__discrete_columns)
    
    # -------------------------------------------------------------------------
    
    @property
    def n_join_keys(self):
        """
        Number of join keys.
        """
        return len(self.__join_key_columns)
    
    # -------------------------------------------------------------------------
    
    @property
    def n_numerical(self):
        """
        Number of numerical columns.
        """
        return len(self.__numerical_columns)
    
    # -------------------------------------------------------------------------

    def n_rows(self):
        """
        Returns the number of rows in the data frame.
        """

        # ------------------------------------------------------
        # Build and send JSON command

        cmd = dict()
        cmd["type_"] = "DataFrame.nrows"
        cmd["name_"] = self.name

        s = comm.send_and_receive_socket(cmd)

        # ------------------------------------------------------
        # Make sure model exists on getml engine

        msg = comm.recv_string(s)

        if msg != "Found!":
            s.close()
            raise Exception(msg)

        # ------------------------------------------------------
        # Receive number of rows from getml engine

        nrows = comm.recv_string(s)

        # ------------------------------------------------------

        s.close()

        return np.int32(nrows)

    # -------------------------------------------------------------------------
    
    @property
    def n_targets(self):
        """
        Number of target columns.
        """
        return len(self.__target_columns)
    
    # -------------------------------------------------------------------------
    
    @property
    def n_time_stamps(self):
        """
        Number of time stamps columns.
        """
        return len(self.__time_stamp_columns)
    
    # -------------------------------------------------------------------------

    def numerical(self, name):
        """
        Returns a handle to a numerical column.

        Args:
            name (str): Name of the column.
        """
        return self.__get_column(name, self.__numerical_columns)

    # -------------------------------------------------------------------------
    
    @property
    def numerical_names(self):
        """
        List of the names of all numerical columns.
        """
        return [col.name for col in self.__numerical_columns]
    
    # -------------------------------------------------------------------------

    def random(self, seed=5849):
        """
        Create random column

        The numbers will uniformly distributed from 0.0 to 1.0.

        Args:
            seed (int)*: Seed used for the random number generator.

        Returns:
            col (Column): Column containing random numbers
        """
        col = columns._VirtualColumn(
            df_name=self.name,
            operator="random",
            operand1=None,
            operand2=None
        )
        col.thisptr["seed_"] = seed
        return col

    # -------------------------------------------------------------------------

    def read_csv(
            self,
            fnames,
            append=True,
            quotechar='"',
            sep=',',
            time_formats=["%Y-%m-%dT%H:%M:%s%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]):
        """
        Read CSV file

        It is assumed that the first line of each CSV file contains the column
        names.

        Args:
            fnames (List[str]): CSV file names to be read.
            append (bool): If a DataFrame already exists, should the file be appended?
            quotechar (str): The character used to wrap strings.
            sep (str): The separator used for separating fields.
            time_formats (str): The formats tried when parsing time stamps.
                Refer to https://pocoproject.org/docs/Poco.DateTimeFormatter.html#9946 for the options.
        """
        # -------------------------------------------
        # Transform paths
        
        fnames_ = [os.path.abspath(_) for _ in fnames]

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "DataFrame.read_csv"
        cmd["name_"] = self.name

        cmd["fnames_"] = fnames_

        cmd["append_"] = append
        cmd["quotechar_"] = quotechar
        cmd["sep_"] = sep
        cmd["time_formats_"] = time_formats

        cmd["categoricals_"] = self.categorical_names
        cmd["discretes_"] = self.discrete_names
        cmd["join_keys_"] = self.join_key_names
        cmd["numericals_"] = self.numerical_names
        cmd["targets_"] = self.target_names
        cmd["time_stamps_"] = self.time_stamp_names

        comm.send(cmd)

        # -------------------------------------------

        return self

    # -------------------------------------------------------------------------

    def refresh(self):
        """
        Aligns meta-information of the DataFrame with the engine.

        This method can be used to avoid encoding conflicts. Note that the
        .load() method automatically calls refresh.
        """

        # ----------------------------------------------------------------------

        cmd = dict()
        cmd["type_"] = "DataFrame.refresh"
        cmd["name_"] = self.name

        s = comm.send_and_receive_socket(cmd)

        msg = comm.recv_string(s)

        s.close()

        if msg[0] != "{":
            raise Exception(msg)

        # ----------------------------------------------------------------------

        encodings = json.loads(msg)

        # ----------------------------------------------------------------------
        # Extract colnames

        if encodings["categorical_"] == '':
            categorical = []
        else:
            categorical = encodings["categorical_"]

        if encodings["discrete_"] == '':
            discrete = []
        else:
            discrete = encodings["discrete_"]

        if encodings["join_keys_"] == '':
            join_keys = []
        else:
            join_keys = encodings["join_keys_"]

        if encodings["numerical_"] == '':
            numerical = []
        else:
            numerical = encodings["numerical_"]

        if encodings["targets_"] == '':
            targets = []
        else:
            targets = encodings["targets_"]

        if encodings["time_stamps_"] == '':
            time_stamps = []
        else:
            time_stamps = encodings["time_stamps_"]

        # ----------------------------------------------------------------------
        # Re-initialize data frame

        self.__init__(
            name=self.name,
            join_keys=join_keys,
            time_stamps=time_stamps,
            categorical=categorical,
            discrete=discrete,
            numerical=numerical,
            targets=targets,
            units=self.units
        )

        # ----------------------------------------------------------------------

        return self

    # -------------------------------------------------------------------------

    def rowid(self):
        """
        Returns a (numerical) column containing the row id, starting with 0.
        """
        return columns._VirtualColumn(
            df_name=self.name,
            operator="rowid",
            operand1=None,
            operand2=None
        )

    # -------------------------------------------------------------------------

    def save(self):
        """Saves the DataFrame on the engine.
        
        To be saved on the engine, it already has to be present
        there. You can use the :meth:`~getml.engine.DataFrame.send`
        function to upload it to the engine.
                
        Returns:
            :class:`~getml.engine.DataFrame`:
                The current instance of the DataFrame class.
        """

        cmd = dict()
        cmd["type_"] = "DataFrame.save"
        cmd["name_"] = self.name

        comm.send(cmd)
        
        return self

    # -------------------------------------------------------------------------

    def rm_categorical(self, name):
        """
        Removes a categorical column from the DataFrame.
        
        Args:
            name (str): Name of the column to be removed.
        """
        self.__rm_col(name, "categorical")

    # -------------------------------------------------------------------------

    def rm_discrete(self, name):
        """
        Removes a discrete column from the DataFrame.

        Args:
            name (str): Name of the column to be removed.
        """
        self.__rm_col(name, "discrete")

    # -------------------------------------------------------------------------

    def rm_join_key(self, name):
        """
        Removes a join key from the DataFrame.

        Args:
            name (str): Name of the column to be removed.
        """
        self.__rm_col(name, "join_key")

    # -------------------------------------------------------------------------

    def rm_numerical(self, name):
        """
        Removes a numerical column from the DataFrame.

        Args:
            name (str): Name of the column to be removed.
        """
        self.__rm_col(name, "numerical")

    # -------------------------------------------------------------------------

    def rm_target(self, name):
        """
        Removes a target column from the DataFrame.

        Args:
            name (str): Name of the column to be removed.
        """
        self.__rm_col(name, "target")

    # -------------------------------------------------------------------------

    def rm_time_stamp(self, name):
        """
        Removes a time stamp column from the DataFrame.

        Args:
            name (str): Name of the column to be removed.
        """
        self.__rm_col(name, "time_stamp")

    # -------------------------------------------------------------------------

    def send(self, data_frame, sock = None):
        """Send data to the getml engine.

        If sock is None, it will call a function to create
        a new socket, use it for the data transfer and close it
        afterwards. If, instead, a socket is provided, it just sends
        all the data but does not close it.

        Args:
            data_frame (pandas.DataFrame): Data Frame that you want to be
                appended to the existing data.
            sock (optional): Socket connecting the Python API with the
                getML engine.

        """

        # ------------------------------------------------------

        if data_frame is not None:
            self.__check_plausibility(data_frame)

        # ------------------------------------------------------
        # Send data frame itself

        cmd = dict()
        cmd["type_"] = "DataFrame"
        cmd["name_"] = self.name

        if sock is None:
            s = comm.send_and_receive_socket(cmd)
        else:
            s = sock
            comm.send_string(s, json.dumps(cmd))

        msg = comm.recv_string(s)

        if msg != "Success!":
            raise Exception(msg)
            
        # ------------------------------------------------------
        # Send individual columns to getml engine

        self.__send_data(data_frame, s)

        # ------------------------------------------------------

        self.__close(s)
        
        if sock is None:
            s.close()

        return self

    # -------------------------------------------------------------------------

    def target(self, name):
        """
        Returns a handle to a target column.

        Args:
            name (str): Name of the column.
        """
        return self.__get_column(name, self.__target_columns)

    # -------------------------------------------------------------------------
    
    @property
    def target_names(self):
        """
        List of the names of all target columns.
        """
        return [col.name for col in self.__target_columns]

    # -------------------------------------------------------------------------

    def time_stamp(self, name):
        """
        Returns a handle to a time stamp column.

        Args:
            name (str): Name of the column.
        """
        return self.__get_column(name, self.__time_stamp_columns)

    # -------------------------------------------------------------------------
    
    @property
    def time_stamp_names(self):
        """
        List of the names of all time stamps.
        """
        return [col.name for col in self.__time_stamp_columns]
    
    # -------------------------------------------------------------------------
    
    def to_csv(self, fname, quotechar='"', sep=','):
        """
        Writes the data frame into a newly created CSV file.

        Args:
            fname (str): The name of the CSV file.
            quotechar (str): The character used to wrap strings.
            sep (str): The separator used for separating fields.
        """
        # ----------------------------------------------------------------------
        # Transform path
        
        fname_ = os.path.abspath(fname)
        
        # ----------------------------------------------------------------------
        # Build command

        cmd = dict()
        cmd["type_"] = "DataFrame.to_csv"
        cmd["name_"] = self.name

        cmd["fname_"] = fname_ 
        cmd["quotechar_"] = quotechar 
        cmd["sep_"] = sep 

        comm.send(cmd)

    # -------------------------------------------------------------------------
    
    def to_db(self, table_name):
        """
        Writes the data frame into a newly created table in the database.

        Args:
            table_name (str): Name of the table to be created. 
                If a table of that name already exists, it will be replaced.
        """
        
        # ----------------------------------------------------------------------
        # Build command

        cmd = dict()
        cmd["type_"] = "DataFrame.to_db"
        cmd["name_"] = self.name

        cmd["table_name_"] = table_name 

        comm.send(cmd)
 
    # -------------------------------------------------------------------------
    
    def to_placeholder(self):
        """
        Generates a placeholder from the data frame.
        """
        self.refresh()
        return Placeholder(
            name=self.name,
            categorical=self.categorical_names,
            discrete=self.discrete_names,
            numerical=self.numerical_names,
            join_keys=self.join_key_names,
            time_stamps=self.time_stamp_names,
            targets=self.target_names
        )
     
    # -------------------------------------------------------------------------

    def where(self, name, condition):
        """
        Creates a new DataFrame as a subselection of this one.

        Args: 
            name (str): Name of the new DataFrame.
            condition (bool): Boolean column indicating the rows you want to select.
        """

        # ----------------------------------------------------------------------
        # Build command

        cmd = dict()
        cmd["type_"] = "DataFrame.where"
        cmd["name_"] = self.name

        cmd["new_df_"] = name
        cmd["condition_"] = condition.thisptr

        comm.send(cmd)

        # ----------------------------------------------------------------------
        # Create handle for new data frame.

        new_df = DataFrame(name)

        return new_df.refresh()

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
