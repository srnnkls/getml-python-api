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

import copy
import datetime
import json
import random
import socket
import string
import sys
import time

import numpy as np
import pandas as pd

import getml.communication as comm
import getml.engine as engine
import getml.loss_functions as loss_functions

from .modutils import _parse_placeholders

# ------------------------------------------------------------------------------


class MultirelModel(object):
    """
    Model for automated feature engineering.

    MultirelModel automates feature engineering for relational data and time series.
    It is based on an efficiently implemented variation of the Multi-Relational
    Decision Tree Learning (MRDTL) algorithm, hence the name.

    Args:
        population (:class:`~getml.models.Placeholder`): Population table (the main table)
        peripheral (List[:class:`~getml.models.Placeholder`]): Peripheral tables
        name (str): Name of the model. Defaults to `None`.
        host (str): Host IP of the getml engine. Defaults to 'localhost'.
        port (int): Port of the getml engine. Defaults to 1708.
        feature_selector (:class:`~getml.predictors`): Predictor used for feature selection
        predictor (:class:`~getml.predictors`): Predictor used for the prediction
        units (dict): Mapping of column names to units. All columns containing
            that column name will be assigned the unit. Columns containing the same
            unit can be directly compared.
        aggregation (List[:class:`~getml.aggregations`]): List of aggregations
            for the algorithm to consider. See module
            :class:`~getml.aggregations` for a list of the aggregations that
            are available. Default: [:class:`~getml.aggregations.Avg`,
            :class:`~getml.aggregations.Count`,
            :class:`~getml.aggregations.Sum`].
        loss_function (:class:`~getml.loss_functions`): Loss function to be
            used to optimize your features. We recommend
            :class:`~getml.loss_functions.SquareLoss` for regression problems and
            :class:`~loss_functions.CrossEntropyLoss` for classification
            problems. Default: :class:`~getml.loss_functions.SquareLoss`.
        use_timestamps (bool): Whether you want to ignore all elements in the peripheral tables where the
            time stamp is greater than the time stamp in the corresponding
            element of the population table.  In other words, this determines
            whether you want add the condition "t2.time_stamp <= t1.time_stamp"
            at the very end of each feature.  It is strongly recommend to keep
            the default value - it is the golden rule of predictive analytics!
            Default: True.
        num_features (int): The number of features you would like to extract. Default: 100.
        share_selected_features (float): The maximum share of features you would like
            to select. Requires you to pass a *feature_selector*. Any features
            with a feature importance of zero will be removed. Therefore, the
            number of features actually selected can be smaller than implied by
            *share_selected_features*. When set to 0.0, no feature selection
            will be conducted. Default: 0.0.
        num_subfeatures (int): The number of subfeatures you would like to
            extract (for snowflake data model only). Default: 10.
        max_length (int): The maximum length a subcondition might have. Multirel will create conditions
            in the form ( condition 1.1 AND condition 1.2 AND condition 1.3 )
            OR ( condition 2.1 AND condition 2.2 AND condition 2.3 ) ... .
            *max_length* determines the maximum number of conditions allowed in
            the brackets. Default: 4.
        min_num_samples (int): This determines the minimum number of samples a
            subcondition should apply to in order for it to be considered. A
            higher number for *min_num_samples* leads to less complex
            statements and less danger of overfitting. Default: 200.
        shrinkage (float) : Multirel works using a gradient-boosting-like
             algorithm. This determines the shrinkage or learning rate of the
             algorithm. Should be between 0.0 and 1.0. A higher shrinkage will
             lead to more danger of overfitting. Default: 0.0.
        sampling_factor (float): Multirel uses a bootstrapping procedure
            (sampling with replacement) to train each of the features. The
            sampling factor is proportional to the share of the samples
            randomly drawn from the population table every time Multirel
            generates a new feature. A lower sampling factor (but still greater
            than 0.0), will lead to less danger of overfitting, less complex
            statements and faster training.  When set to 1.0 (the default
            value), the sampling rate will be set such that roughly 2,000
            samples are drawn from the population table. If the population
            table contains less than 2,000 samples, it will use standard
            bagging.  When set to 0.0, there will be no sampling at all.
            Default: 1.0,
        round_robin (bool): If True, the Multirel picks a different aggregation
            every time a new feature is generated. Default: False.
        share_aggregations (float): Every time a new feature is generated, the
            aggregation will be taken from a random subsample of possible
            aggregations and values to be aggregated. *share_aggregations*
            determines the size of that subsample. Must be between 0.0 and 1.0.
            Only relevant when round_robin is False. Default: 0.25.
        share_conditions (float): Every time a new column is tested for
            applying conditions, the column might be skipped at random.
            *share_conditions* determines the probability that a column will
            *not* be skipped. Must be between 0.0 and 1.0. Default: 1.0.
        allow_sets (bool): Multirel can summarize different categories into a
            sets for producing conditions. In the SQL statements these sets
            might look like this: t2.category IN ( 'value_1', 'value_2', ... ).
            This can be very powerful, but it can also produce features that
            are hard to read and might be prone to overfitting when the
            *sampling_factor* is too low. Default: True.
        delta_t (float): Frequency with which lag variables will be explored in
            a time series setting. When set to 0.0, there will be to lag
            variables. Default: 0.0.
        include_categorical (bool): Whether you want to pass the categorical
            columns from the population table to the predictors. Default:
            False.
        grid_factor (float): Multirel will try a grid of critical values for
            your numerical features. A higher *grid_factor* will lead to more
            critical values being considered. This can increase the training
            time, but also lead to more accurate features. Default: 1.0.
        regularization (float): Regularizes your features. A higher
            *regularization* will lead to less complex features and less danger
            of overfitting. A *regularization* of 1.0 is very strong
            regularization that allows no conditions. Default: 0.0.
        seed (int): Seed used for the random number generator that underlies
            the sampling procedure. Default: 5489
        num_threads (int): Number of threads used for feature generation. If
            set to zero or a negative value, the number of threads will be
            determined automatically by the getml engine. Default: 0.
        send (True): If True, the Model will be automatically sent to the getml
            engine without you having to explicitly call .send(). Default:
            False.
    """    

    # -------------------------------------------------------------------------

    def __init__(self, name=None, **params):

        self.name = name or \
            datetime.datetime.now().isoformat().split(".")[0].replace(':', '-') + "-multirel"

        self.predictor = None

        self.thisptr = dict()
        self.thisptr["type_"] = "MultirelModel"
        self.thisptr["name_"] = self.name

        self.params = {
            'population': None,
            'peripheral': None,
            'host': 'localhost',
            'port': 1708,
            'units': dict(),
            'send': False,
            'aggregation': ["AVG", "COUNT", "SUM"],
            'loss_function': loss_functions.SquareLoss().thisptr["type_"],
            'use_timestamps':  True,
            'num_features': 100,
            'share_selected_features': 0.0,
            'num_subfeatures': 10,
            'max_length': 4,
            'min_num_samples': 200,
            'shrinkage': 0.0,
            'sampling_factor': 1.0,
            'round_robin': False,
            'session_name': "",
            'share_aggregations': 0.25,
            'share_conditions': 1.0,
            'allow_sets': True,
            'delta_t': 0.0,
            'include_categorical': False,
            'grid_factor': 1.0,
            'regularization': 0.0,
            'seed': 5489,
            'num_threads': 0,
            'feature_selector': None,
            'predictor': None
        }

        self.set_params(params)

        if self.params['send']:
            self.send()

    # -------------------------------------------------------------------------

    def __close(self, s):

        cmd = dict()
        cmd["type_"] = "MultirelModel.close"
        cmd["name_"] = self.name

        comm.send_cmd(s, json.dumps(cmd))

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
        cmd["type_"] = "MultirelModel.fit"
        cmd["name_"] = self.name

        cmd["peripheral_names_"] = [df.name for df in peripheral_data_frames]
        cmd["population_name_"] = population_data_frame.name

        comm.send_cmd(s, json.dumps(cmd))

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
            self.__close(s)
        elif "has already been fitted" in msg:
            print(msg)
            print("")
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
                        units=self.params['units'],
                        host=self.params['host'],
                        port=self.params['port']
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
                units=self.params["units"],
                host=self.params["host"],
                port=self.params["port"]
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
        # Establish communication with getml engine

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params["host"], self.params["port"]))

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.save"
        cmd["name_"] = self.name

        comm.send_cmd(
            s,
            json.dumps(cmd)
        )

        # -------------------------------------------
        # Make sure everything went well and close
        # connection

        msg = comm.recv_string(s)

        s.close()

        if msg != "Success!":
            raise Exception(msg)

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
        cmd["type_"] = "MultirelModel.score"
        cmd["name_"] = self.name

        #cmd["num_threads_"] = self.params["num_threads"]

        # ----------------------------------------------------------------------
        # Establish connection with the getml engine and send command

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params['host'], self.params['port']))

        comm.send_cmd(s, json.dumps(cmd))

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
        # Prepare the command for the getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.transform"
        cmd["name_"] = self.name

        cmd["score_"] = score
        cmd["predict_"] = predict

        cmd["peripheral_names_"] = [df.name for df in peripheral_data_frames]
        cmd["population_name_"] = population_data_frame.name

        cmd["table_name_"] = table_name 

        comm.send_cmd(s, json.dumps(cmd))

        # -----------------------------------------------------
        # Do the actual transformation

        msg = comm.recv_string(s)

        if msg == "Success!":
            if table_name == "":
                y_hat = comm.recv_matrix(s)
            else:
                y_hat = None
        else:
            raise Exception(msg)

        # -----------------------------------------------------

        return y_hat

    # -------------------------------------------------------------------------

    def copy(self, other):
        """
        Copies the parameters and hyperparameters from another model.

        Args:
            other (:class:`getml.models.MultirelModel`): The other model.
        """

        # -------------------------------------------
        # Establish communication with getml engine

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params["host"], self.params["port"]))

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.copy"
        cmd["name_"] = self.name
        cmd["other_"] = other.name

        comm.send_cmd(
            s,
            json.dumps(cmd)
        )

        # -------------------------------------------
        # Make sure everything went well and close
        # connection

        msg = comm.recv_string(s)

        s.close()

        if msg != "Success!":
            raise Exception(msg)

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
        # Establish communication with getml engine

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params["host"], self.params["port"]))

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.delete"
        cmd["name_"] = self.name
        cmd["mem_only_"] = mem_only

        comm.send_cmd(
            s,
            json.dumps(cmd)
        )

        # -------------------------------------------
        # Make sure everything went well and close
        # connection

        msg = comm.recv_string(s)

        s.close()

        if msg != "Success!":
            raise Exception(msg)

    # -------------------------------------------------------------------------

    def fit(
        self,
        population_table,
        peripheral_tables
    ):
        """
        Fits the model.

        Args:
            population_table (:class:`pandas.DataFrame` or
                :class:`~getml.engine.DataFrame`): Population table containing the target
            peripheral_tables (List[:class:`pandas.DataFrame` or
                :class:`~getml.engine.DataFrame`]): Peripheral tables
        """

        # -----------------------------------------------------
        # Prepare the command for the getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.fit"
        cmd["name_"] = self.name

        # -----------------------------------------------------
        # Send command to engine and make sure that model has
        # been found

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params['host'], self.params['port']))

        comm.send_cmd(s, json.dumps(cmd))

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
        # Establish communication with getml engine

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params["host"], self.params["port"]))

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.load"
        cmd["name_"] = self.name

        comm.send_cmd(
            s,
            json.dumps(cmd)
        )

        # -------------------------------------------
        # Make sure everything went well and close
        # connection

        msg = comm.recv_string(s)

        s.close()

        if msg != "Success!":
            raise Exception(msg)

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
        Predictions generated by the model.
        
        Requires that you have passed a predictor.

        Args:  
            population_table (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Population table. Targets will be ignored
            peripheral_tables (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Peripheral tables
            table_name (str): If not an empty string, the resulting features
                 will be written into the database, instead of returning them.
        """
        # -----------------------------------------------------
        # Prepare the command for the getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.transform"
        cmd["name_"] = self.name

        # -----------------------------------------------------
        # Send command to engine and make sure that model has
        # been found

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params['host'], self.params['port']))

        comm.send_cmd(s, json.dumps(cmd))

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
            predict=True, 
            table_name=table_name
        )

        self.__close(s)

        s.close()

        return y_hat

    # -------------------------------------------------------------------------

    def refresh(self):
        """
        Refreshes the hyperparameters and placeholders in Python based on a 
        model already loaded in the engine.
        """

        # -------------------------------------------
        # Establish communication with getml engine

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params["host"], self.params["port"]))

        # -------------------------------------------
        # Send JSON command to getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.refresh"
        cmd["name_"] = self.name

        comm.send_cmd(
            s,
            json.dumps(cmd)
        )

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
            population_table (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Population table. Targets will be ignored
            peripheral_tables (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Peripheral tables
        """
        # -----------------------------------------------------
        # Prepare the command for the getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.transform"
        cmd["name_"] = self.name

        # -----------------------------------------------------
        # Send command to engine and make sure that model has
        # been found

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params['host'], self.params['port']))

        comm.send_cmd(s, json.dumps(cmd))

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

        yhat = self.__transform(
            peripheral_data_frames, population_data_frame, s, predict=True, score=True)

        # ----------------------------------------------------------------------
        # Get targets

        y = pd.DataFrame()

        for colname in population_data_frame.target_names:
            y[colname] = population_data_frame.target(colname).get(s).ravel()

        y = y.values

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
        Send this MultirelModel to the getml engine.
        """

        # -------------------------------------------
        # Establish communication with getml engine

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params["host"], self.params["port"]))

        # -------------------------------------------
        # Send own JSON command to getml engine

        if self.params["population"] is None:
            raise Exception("Population cannot be None!")

        if self.params["peripheral"] is None:
            raise Exception("Peripheral cannot be None!")

        cmd = dict()
        cmd["name_"] = self.name
        cmd["type_"] = "MultirelModel"
        cmd["population_"] = self.params["population"].thisptr
        cmd["peripheral_"] = [per.thisptr["name_"]
                              for per in self.params["peripheral"]]
        cmd["hyperparameters_"] = self.__make_hyperparameters()

        comm.send_cmd(
            s,
            json.dumps(cmd)
        )

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

    def set_params(self, params=None, **kwargs):
        """
        Sets the hyperparameters of the model.

        Args: 
            params (dict): Hyperparameters that were returned by :func:`~getml.models.MultirelModel.get_params`.
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
                try:
                    self.params[key] = value._getml_thisptr()
                except:
                    self.params[key] = value

            elif key == "feature_selector":
                try:
                    self.params[key] = value._getml_thisptr()
                except:
                    self.params[key] = value

            else:
                self.params[key] = value

        return self

    # -------------------------------------------------------------------------

    def set_peripheral_tables(self, peripheral_tables):
        """
        Sets the peripheral tables.

        This is very useful for establishing a pipeline.

        Args:
            peripheral_tables (List[:class:`pandas.DataFrame` or
                 :class:`~getml.engine.DataFrame`]): Peripheral tables
        """

        self.params['peripheral_tables'] = peripheral_tables

    # -------------------------------------------------------------------------

    def to_sql(self):
        """
        Extracts the SQL statements underlying the trained model.
        """

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params['host'], self.params['port']))

        # ------------------------------------------------------
        # Build and send JSON command

        cmd = dict()
        cmd["type_"] = "MultirelModel.to_sql"
        cmd["name_"] = self.name

        comm.send_cmd(s, json.dumps(cmd))

        # ------------------------------------------------------
        # Make sure model exists on getml engine

        msg = comm.recv_string(s)

        if msg != "Found!":
            raise Exception(msg)

        # ------------------------------------------------------
        # Receive SQL code from getml engine

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
        Returns the features learned by the model.

        Args:  
            population_table (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Population table. Targets will be ignored
            peripheral_tables (List[:class:`pandas.DataFrame` or :class:`~getml.engine.DataFrame`]):
                Peripheral tables
            table_name (str): If not an empty string, the resulting features
                will be written into the database, instead of returning them.
        """

        # -----------------------------------------------------
        # Prepare the command for the getml engine

        cmd = dict()
        cmd["type_"] = "MultirelModel.transform"
        cmd["name_"] = self.name

        # -----------------------------------------------------
        # Send command to engine and make sure that model has
        # been found

        s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        s.connect((self.params['host'], self.params['port']))

        comm.send_cmd(s, json.dumps(cmd))

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
