import json

import getml.communication as comm

# -----------------------------------------------------------------------------------------

class Container(object):
    """
    Base object not meant to be called directly
    """

    # -------------------------------------------------------------------------

    def __init__(self):

        self.colnames = None
        self.units = None

        self.thisptr = dict()

    # -------------------------------------------------------------------------

    def send(self, numpy_array, s):
        """
        Sends the object to the engine, data taken from a numpy array.

        Args:
            numpy_array (:class:`numpy.ndarray`): Number of columns should match the number of columns of the object itself.
            s: Socket
        """
        
        # -------------------------------------------
        # Send own JSON command to getml engine

        comm.send_string(s, json.dumps(self.thisptr))

        # -------------------------------------------
        # Send data to getml engine

        if self.thisptr["type_"] == "CategoricalColumn":
            comm.send_categorical_matrix(s, numpy_array)

        elif self.thisptr["type_"] == "Column":
            comm.send_matrix(s, numpy_array)

        # -------------------------------------------
        # Make sure everything went well

        msg = comm.recv_string(s)

        if msg != "Success!":
            raise Exception(msg)

        # -------------------------------------------

        if len(numpy_array.shape) > 1:
            self.colnames = self.colnames or [
                "column_" + str(i + 1) for i in range(numpy_array.shape[1])
            ]

    # -------------------------------------------------------------------------

    def set_unit(self, unit):
        """
        Sets the unit of the column.

        Args:
            unit: The new unit.
        """

        # -------------------------------------------
        # Build command string

        cmd = dict()

        cmd.update(self.thisptr)

        cmd["unit_"] = unit

        cmd["type_"] += ".set_unit"

        # -------------------------------------------
        # Send JSON command to engine

        comm.send(cmd)

        # -------------------------------------------
        # Store the new unit

        self.thisptr["unit_"] = unit

# -----------------------------------------------------------------------------
