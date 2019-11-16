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
import random
import string
import time

import pandas as pd

import getml.communication as comm
import getml.engine as engine
import getml.loss_functions as loss_functions

from .modutils import _parse_placeholders

# ------------------------------------------------------------------------------


class RelboostModel(object):
    """
    Generalization of the XGBoost algorithm to relational data.
    
    RelboostModel automates feature engineering for relational data and time series.
    It is based on a generalization of the XGBoost algorithm to relational data, 
    hence the name.
    
    Args:
        population (:class:`~getml.models.Placeholder`): Population table (the main table)
        peripheral (List[:class:`~getml.models.Placeholder`]): Peripheral tables
        name (str): Name of the model. Defaults to `None`.
        feature_selector (:class:`~getml.predictors`): Predictor used for feature selection
        predictor (:class:`~getml.predictors`): Predictor used for the prediction
        units (dict): Mapping of column names to units. All columns containing
            that column name will be assigned the unit. Columns containing the same
            unit can be directly compared.
        loss_function (:class:`~getml.loss_functions`): Loss function to be
            used to optimize your features. We recommend
            :class:`~getml.loss_functions.SquareLoss` for regression problems and
            :class:`~loss_functions.CrossEntropyLoss` for classification
            problems. Default: :class:`~getml.loss_functions.SquareLoss`.
        delta_t (float): Frequency with which lag variables will be explored in a time
            series setting. When set to 0.0, then there will be to lag variables. 
            Default: 0.0.
        gamma (float): Minimum improvement required for a split. Default: 0.0.
        include_categorical (bool): Whether you want to pass the categorical columns from the population table to the predictors.
            Default: False.
        max_depth (int): Maximum depth of the trees. Default: 3.
        num_features (int): The number of features to be trained. Default: 100.
        share_selected_features (float): The maximum share of features you would like to
            select. Requires you to pass a *feature_selector*. Any features with a 
            feature importance of zero will be removed. Therefore,
            the number of features actually selected can be smaller than implied by
            *share_selected_features*. When set to 0.0,
            no feature selection will be conducted. Default: 0.0. 
        num_subfeatures (int): The number of subfeatures you would like to
            extract (for snowflake data model only). Default: 100.
        reg_lambda (float): L2 regularization on the weights. This is probably one of the most important hyperparameters
            in the *RelboostModel*. Default: 0.01.
        seed (int): The seed used for initializing the random number generator. Default: 5843.
        shrinkage (float): Learning rate to be used for the boosting algorithm. Default: 0.3.
        silent (bool): Controls the logging during training. Default: False.
        subsample (float): Subsample ratio during training. Set to 0.0 for no subsampling. Default: 1.0.
        target_num (int): Signifies which of the targets to use, since RelboostModel does not
            support multiple targets. Default: 0.
        use_timestamps (bool): Whether you want to ignore all elements in the peripheral tables where the
            time stamp is greater than the time stamp in the corresponding element of the population table.
            In other words, this determines whether you want add the condition "t2.time_stamp <= t1.time_stamp" at
            the very end of each feature.
            We strongly recommend that you keep the default value - it is the golden rule of predictive analytics!
            Default: True.
    """
    # -------------------------------------------------------------------------

    def __init__(self, name=None, **params):

        self.name = name or \
            datetime.datetime.now().isoformat().split(".")[0].replace(':', '-') + "-relboost"

        self.feature_selector = None
        self.predictor = None

        self.thisptr = dict()
        self.thisptr["type_"] = "RelboostModel"
        self.thisptr["name_"] = self.name

        self.params = {
            'population': None,
            'peripheral': None,
            'units': dict(),
            'send': False,
            'delta_t': 0.0,
            'feature_selector': None,
            'gamma': 0.0,
            'include_categorical': False,
            'loss_function': "SquareLoss",
            'max_depth': 3,
            'min_num_samples': 200,
            'num_features': 100,
            'num_subfeatures': 100,
            'session_name': "",
            'share_selected_features': 0.0,
            'num_threads': 0,
            'predictor': None,
            'reg_lambda': 0.01,
            'sampling_factor': 1.0,
            'shrinkage': 0.3,
            'seed': 5843,
            'silent': False,
            'target_num': 0,
            'use_timestamps': True
        }

        self.set_params(params)

        if self.params['send']:
            self.send()

    # -------------------------------------------------------------------------

    def __close(self, s):

        cmd = dict()
        cmd["type_"] = "RelboostModel.close"
        cmd["name_"] = self.name

        comm.send_string(s, json.dumps(cmd))

        msg = comm.recv_string(s)

        if msg != "Success!":
            raise Exception(msg)

    # -------------------------------------------------------------------------

    def __fit(
        self,
        peripheral_data_frames,
        population_data_frame,
        s
    ):

        # -----------------------------------------------------
        # Send the complete fit command.

        cmd = dict()
        cmd["type_"] = "RelboostModel.fit"
        cmd["name_"] = self.name

        cmd["peripheral_names_"] = [df.name for df in peripheral_data_frames]
        cmd["population_name_"] = population_data_frame.name

        comm.send_string(s, json.dumps(cmd))

        # -----------------------------------------------------
        # Do the actual fitting

        begin = time.time()

        print("Loaded data. Features are now being trained...")

        msg = comm.recv_string(s)

        end = time.time()

        # ----------------------------------------------------------------------
        # Print final message

        if "Trained" in msg:
            print(msg)
            self.__print_time_taken(begin, end, "Time taken: ")
        else:
            raise Exception(msg)

    # -------------------------------------------------------------------------

    def __load_peripheral_tables(self, peripheral_tables, s):

        peripheral_data_frames = []

        for i, peripheral_table in enumerate(peripheral_tables):
            if type(peripheral_table) == engine.DataFrame:

                peripheral_data_frames.append(peripheral_table)

            else:

                categorical_peripheral = [
                    per.thisptr["categorical_"] for per in self.params["peripheral"]
                ]

                discrete_peripheral = [
                    per.thisptr["discrete_"] for per in self.params["peripheral"]
                ]

                numerical_peripheral = [
                    per.thisptr["numerical_"] for per in self.params["peripheral"]
                ]

                join_keys_peripheral = [
                    per.thisptr["join_keys_"] for per in self.params["peripheral"]
                ]

                names_peripheral = [
                    per.thisptr["name_"] for per in self.params["peripheral"]
                ]

                time_stamps_peripheral = [
                    per.thisptr["time_stamps_"] for per in self.params["peripheral"]
                ]

                peripheral_data_frames.append(
                    engine.DataFrame(
                        name=self.__make_random_name(),
                        join_keys=join_keys_peripheral[i],
                        time_stamps=time_stamps_peripheral[i],
                        categorical=categorical_peripheral[i],
                        discrete=discrete_peripheral[i],
                        numerical=numerical_peripheral[i],
                        targets=[],
                        units=self.params['units']
                    )
                )

                peripheral_data_frames[i].send(
                    data_frame=peripheral_table,
                    sock=s
                )

        return peripheral_data_frames

    # -------------------------------------------------------------------------

    def __load_population_table(self, population_table, targets, s):

        if type(population_table) == engine.DataFrame:

            population_data_frame = population_table

        else:

            population_data_frame = engine.DataFrame(
                name=self.__make_random_name(),
                join_keys=self.params["population"].thisptr["join_keys_"],
                time_stamps=self.params["population"].thisptr["time_stamps_"],
                categorical=self.params["population"].thisptr["categorical_"],
                discrete=self.params["population"].thisptr["discrete_"],
                numerical=self.params["population"].thisptr["numerical_"],
                targets=targets,
                units=self.params["units"]
            )

            population_data_frame.send(
                data_frame=population_table,
                sock=s
            )

        return population_data_frame

    # -------------------------------------------------------------------------

    def __make_hyperparameters(self):

        hyperparameters = dict()

        for key, value in self.params.items():

            if key == "population" or key == "peripheral":
                continue

            hyperparameters[key + "_"] = value

        return hyperparameters

    # -------------------------------------------------------------------------

    def __make_random_name(self):
        return "temp-" + ''.join(
            random.choice(string.ascii_letters) for i in range(15)
        )

    # -------------------------------------------------------------------------

    def __print_time_taken(self, begin, end, msg):
        seconds = end - begin

        hours = int(seconds / 3600)
        seconds -= float(hours * 3600)

        minutes = int(seconds / 60)
        seconds -= float(minutes * 60)

        seconds = round(seconds, 6)

        print(
            msg + str(hours) + "h:" +
            str(minutes) + "m:" + str(seconds)
        )
        print("")

    # -------------------------------------------------------------------------

    def __repr__(self):
        return self.get_params().__repr__()

    # -------------------------------------------------------------------------

    def __save(self):
        """
        Saves the model as a JSON file.
        """

        # -------------------------------------------
        # Send JSON command to getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.save"
        cmd["name_"] = self.name

        comm.send(cmd)

    # -------------------------------------------------------------------------

    def __score(self, yhat, y):
        """
        Returns the score for a set of predictions.
        
        **yhat**: Predictions.

        **y**: Targets.
        """

        # ----------------------------------------------------------------------
        # Build the cmd string

        cmd = dict()
        cmd["type_"] = "RelboostModel.score"
        cmd["name_"] = self.name

        # ----------------------------------------------------------------------
        # Establish connection with the getML engine and send command

        s = comm.send_and_receive_socket(cmd)

        msg = comm.recv_string(s)

        if msg != "Found!":
            raise Exception(msg)

        # ----------------------------------------------------------------------
        # Send data
        
        comm.send_matrix(s, yhat)

        comm.send_matrix(s, y)

        msg = comm.recv_string(s)

        # ----------------------------------------------------------------------
        # Ensure success, receive scores

        if msg != "Success!":
            raise Exception(msg)

        scores = comm.recv_string(s)

        s.close()

        # ----------------------------------------------------------------------

        return json.loads(scores)

    # -------------------------------------------------------------------------

    def __transform(
        self,
        peripheral_data_frames,
        population_data_frame,
        s,
        score=False,
        predict=False,
        table_name=""
    ):

        # -----------------------------------------------------
        # Prepare the command for the getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.transform"
        cmd["name_"] = self.name

        cmd["score_"] = score
        cmd["predict_"] = predict

        cmd["peripheral_names_"] = [df.name for df in peripheral_data_frames]
        cmd["population_name_"] = population_data_frame.name

        cmd["table_name_"] = table_name 

        comm.send_string(s, json.dumps(cmd))

        # -----------------------------------------------------
        # Do the actual transformation

        msg = comm.recv_string(s)

        if msg == "Success!":
            if table_name == "":
                yhat = comm.recv_matrix(s)
            else:
                yhat = None
        else:
            raise Exception(msg)

        # -----------------------------------------------------

        return yhat

    # -------------------------------------------------------------------------

    def copy(self, other):
        """
        Copies the parameters and hyperparameters from another model.

        Args:
            other (:class:`getml.models.RelboostModel`): The other model.
        """

        # -------------------------------------------
        # Send JSON command to getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.copy"
        cmd["name_"] = self.name
        cmd["other_"] = other.name

        comm.send(cmd)

        # -------------------------------------------

        self.refresh()

    # -------------------------------------------------------------------------

    def delete(self, mem_only=False):
        """
        Deletes the model from the engine.

        Args:
            mem_only (bool): If True, then the data frame will be deleted from
                memory only, but not from disk. Default: False.
        """

        # -------------------------------------------
        # Send JSON command to getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.delete"
        cmd["name_"] = self.name
        cmd["mem_only_"] = mem_only

        comm.send(cmd)

    # -------------------------------------------------------------------------

    def fit(
        self,
        population_table,
        peripheral_tables
    ):
        """
        Fits the model.

        Args:
            population_table (:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`): 
                Population table containing the target.
            peripheral_tables (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Peripheral tables.
                The peripheral tables have to be passed in the exact same order as their
                corresponding placeholders!
        """

        # -----------------------------------------------------
        # Prepare the command for the getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.fit"
        cmd["name_"] = self.name

        # -----------------------------------------------------
        # Send command to engine and make sure that model has
        # been found

        s = comm.send_and_receive_socket(cmd)

        msg = comm.recv_string(s)

        if msg != "Found!":
            raise Exception(msg)

        # ----------------------------------------------------------------------
        # Load peripheral tables

        peripheral_tables = peripheral_tables or self.params['peripheral_tables']

        peripheral_data_frames = self.__load_peripheral_tables(
            peripheral_tables,
            s
        )

        # ----------------------------------------------------------------------
        # Load population table

        targets = self.params['population'].thisptr["targets_"]

        population_data_frame = self.__load_population_table(
            population_table,
            targets,
            s
        )

        # ----------------------------------------------------------------------
        # Call the __fit(...) method, which does the actual fitting.

        self.__fit(peripheral_data_frames, population_data_frame, s)

        # ----------------------------------------------------------------------

        self.__close(s)

        s.close()

        self.__save()

        return self.refresh()

    # -------------------------------------------------------------------------

    def get_params(self):
        """
        Returns the hyperparameters of the model.
        """
        return self.params

    # -------------------------------------------------------------------------

    def get_predictor(self):
        """
        Returns the predictor of the model.
        """
        return self.params["predictor"]

    # -------------------------------------------------------------------------

    def load(self):
        """
        Loads the model from a JSON file.
        """

        # -------------------------------------------
        # Send JSON command to getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.load"
        cmd["name_"] = self.name

        comm.send(cmd)

        # -------------------------------------------

        return self.refresh()

    # -------------------------------------------------------------------------

    def predict(
        self,
        population_table,
        peripheral_tables=None,
        table_name=""
    ):
        """
        Returns the predictions generated by the model or writes them into a data base.
        
        Requires that you have passed a predictor.

        Args:  
            population_table (:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`):
                Population table. Targets will be ignored
            peripheral_tables (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Peripheral tables.
                The peripheral tables have to be passed in the exact same order as their
                corresponding placeholders!
            table_name (str): If not an empty string, the resulting features
                 will be written into the data base, instead of returning them.
        """

        # -----------------------------------------------------
        # Prepare the command for the getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.transform"
        cmd["name_"] = self.name

        # -----------------------------------------------------
        # Send command to engine and make sure that model has
        # been found

        s = comm.send_and_receive_socket(cmd)

        msg = comm.recv_string(s)

        if msg != "Found!":
            raise Exception(msg)

        # ----------------------------------------------------------------------
        # Load peripheral tables

        peripheral_tables = peripheral_tables or self.params['peripheral_tables']

        peripheral_data_frames = self.__load_peripheral_tables(
            peripheral_tables,
            s
        )

        # ----------------------------------------------------------------------
        # Load population table

        if type(population_table) == engine.DataFrame:
            targets = []
        else:
            targets = [
                elem for elem in self.params['population'].thisptr["targets_"]
                if elem in population_table.columns
            ]

        population_data_frame = self.__load_population_table(
            population_table,
            targets,
            s
        )

        # ----------------------------------------------------------------------
        # Get predictions as numpy array

        yhat = self.__transform(
            peripheral_data_frames,
            population_data_frame,
            s,
            predict=True,
            score=False,
            table_name=table_name
        )

        # ----------------------------------------------------------------------
        # Close connection.

        self.__close(s)

        s.close()

        # ----------------------------------------------------------------------

        return yhat
    
    # -------------------------------------------------------------------------

    def refresh(self):
        """
        Refreshes the hyperparameters and placeholders in Python based on a 
        model already loaded in the engine.
        """

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.refresh"
        cmd["name_"] = self.name

        s = comm.send_and_receive_socket(cmd)

        # -------------------------------------------
        # Make sure everything went well and close
        # connection

        msg = comm.recv_string(s)

        if msg[0] != '{':
          raise Exception(msg)

        s.close()

        # -------------------------------------------
        # Parse results.

        json_obj = json.loads(msg)

        self.set_params(json_obj["hyperparameters_"])

        self.params = _parse_placeholders(
            json_obj, 
            self.params
        )

        return self

    # -------------------------------------------------------------------------

    def score(
        self,
        population_table,
        peripheral_tables=None
    ):
        """
        Calculates scores for the model.

        Args:  
            population_table (:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`):
                Population table. Targets will be ignored
            peripheral_tables (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Peripheral tables.
                The peripheral tables have to be passed in the exact same order as their
                corresponding placeholders!
        """
        # -----------------------------------------------------
        # Prepare the command for the getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.transform"
        cmd["name_"] = self.name

        # -----------------------------------------------------
        # Send command to engine and make sure that model has
        # been found

        s = comm.send_and_receive_socket(cmd)

        msg = comm.recv_string(s)

        if msg != "Found!":
            raise Exception(msg)

        # ----------------------------------------------------------------------
        # Load peripheral tables

        peripheral_tables = peripheral_tables or self.params['peripheral_tables']

        peripheral_data_frames = self.__load_peripheral_tables(
            peripheral_tables,
            s
        )

        # ----------------------------------------------------------------------
        # Load population table

        if type(population_table) == engine.DataFrame:
            targets = []
        else:
            targets = [
                elem for elem in self.params['population'].thisptr["targets_"]
                if elem in population_table.columns
            ]

        population_data_frame = self.__load_population_table(
            population_table,
            targets,
            s
        )

        # ----------------------------------------------------------------------
        # Get predictions as numpy array

        yhat = self.__transform(
            peripheral_data_frames,
            population_data_frame,
            s,
            predict=True,
            score=True)

        # ----------------------------------------------------------------------
        # Get targets

        colname = population_data_frame.target_names[self.params["target_num"]] 

        y = population_data_frame.target(colname).get(s).ravel()

        # ----------------------------------------------------------------------
        # Close connection.

        self.__close(s)

        s.close()

        # ----------------------------------------------------------------------
        # Do the actual scoring.

        scores = self.__score(yhat, y)

        # ----------------------------------------------------------------------

        self.__save()

        return scores

    # -------------------------------------------------------------------------

    def send(self):
        """
        Send this RelboostModel to the getml engine.
        """

        # -------------------------------------------
        # Send own JSON command to getML engine

        if self.params["population"] is None:
            raise Exception("Population cannot be None!")

        if self.params["peripheral"] is None:
            raise Exception("Peripheral cannot be None!")

        cmd = dict()
        cmd["name_"] = self.name
        cmd["type_"] = "RelboostModel"
        cmd["population_"] = self.params["population"].thisptr
        cmd["peripheral_"] = [per.thisptr["name_"]
                              for per in self.params["peripheral"]]
        cmd["hyperparameters_"] = self.__make_hyperparameters()

        comm.send(cmd)

        # -------------------------------------------

        return self

    # -------------------------------------------------------------------------

    def set_params(self, params=None, **kwargs):
        """
        Sets the hyperparameters of the model.

        Args: 
            params (dict): Hyperparameters that were returned by :func:`~getml.models.RelboostModel.get_params`.
        """

        if params is not None:
            items = params.items()
        else:
            items = kwargs.items()

        valid_params = self.get_params()

        for key, value in items:

            if key[-1] == "_":
                key = key[:-1]

            if key not in valid_params:
                raise ValueError(
                    'Invalid parameter %s. ' %
                    (key)
                )

            if key == "loss_function":
                try:
                    self.params[key] = value.thisptr["type_"]
                except:
                    self.params[key] = value

            elif key == "predictor":
                self.predictor = value
                try:
                    self.params[key] = value._getml_thisptr()
                except:
                    self.params[key] = None

            elif key == "feature_selector":
                self.feature_selector = value
                try:
                    self.params[key] = value._getml_thisptr()
                except:
                    self.params[key] = None

            else:
                self.params[key] = value

        return self

    # -------------------------------------------------------------------------

    def set_peripheral_tables(self, peripheral_tables):
        """
        Sets the peripheral tables.

        This is very useful for establishing a pipeline.

        Args:
            peripheral_tables (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]): 
                Peripheral tables.
                The peripheral tables have to be passed in the exact same order as their
                corresponding placeholders!
        """

        self.params['peripheral_tables'] = peripheral_tables

    # -------------------------------------------------------------------------

    def to_sql(self):
        """
        Extracts the SQL statements underlying the trained model.
        """

        # ------------------------------------------------------
        # Build and send JSON command

        cmd = dict()
        cmd["type_"] = "RelboostModel.to_sql"
        cmd["name_"] = self.name

        s = comm.send_and_receive_socket(cmd)

        # ------------------------------------------------------
        # Make sure model exists on getML engine

        msg = comm.recv_string(s)

        if msg != "Found!":
            raise Exception(msg)

        # ------------------------------------------------------
        # Receive SQL code from getML engine

        sql = comm.recv_string(s)

        # ------------------------------------------------------

        s.close()

        return sql

    # -------------------------------------------------------------------------

    def transform(
        self,
        population_table,
        peripheral_tables=None,
        table_name=""
    ):
        """
        Returns the features learned by the model or writes them into a data base.

        Args:  
            population_table (:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`):
                Population table. Targets will be ignored.
            peripheral_tables (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Peripheral tables.
                The peripheral tables have to be passed in the exact same order as their
                corresponding placeholders!
            table_name (str): If not an empty string, the resulting features
                will be written into the data base, instead of returning them.
        """

        # -----------------------------------------------------
        # Prepare the command for the getML engine

        cmd = dict()
        cmd["type_"] = "RelboostModel.transform"
        cmd["name_"] = self.name

        # -----------------------------------------------------
        # Send command to engine and make sure that model has
        # been found

        s = comm.send_and_receive_socket(cmd)

        msg = comm.recv_string(s)

        if msg != "Found!":
            raise Exception(msg)

        # ----------------------------------------------------------------------
        # Load peripheral tables

        peripheral_tables = peripheral_tables or self.params['peripheral_tables']

        peripheral_data_frames = self.__load_peripheral_tables(
            peripheral_tables,
            s
        )

        # ----------------------------------------------------------------------
        # Load population table

        if type(population_table) == engine.DataFrame:
            targets = []
        else:
            targets = [
                elem for elem in self.params['population'].thisptr["targets_"]
                if elem in population_table.columns
            ]

        population_data_frame = self.__load_population_table(
            population_table,
            targets,
            s
        )

        # ----------------------------------------------------------------------
        # Call the predict function to get features as numpy array

        y_hat = self.__transform(
            peripheral_data_frames, 
            population_data_frame, 
            s, 
            table_name=table_name
        )

        self.__close(s)

        s.close()

        return y_hat

# ------------------------------------------------------------------------------
