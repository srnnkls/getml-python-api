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

from getml.placeholder import Placeholder

# -----------------------------------------------------------------------------

def _extract_schema(schema, thisptr):
    
    thisptr["categorical_"] = schema["categoricals_"] 

    thisptr["discrete_"] = schema["discretes_"] 

    thisptr["join_keys_"] = schema["join_keys_"] 

    thisptr["numerical_"] = schema["numericals_"]

    thisptr["targets_"] = schema["targets_"] 

    thisptr["time_stamps_"] = schema["time_stamps_"]

    return thisptr
    
# -----------------------------------------------------------------------------

def _parse_placeholders(json_obj, params):

    if len(json_obj["peripheral_"]) != len(json_obj["peripheral_schema_"]):
        raise Exception("peripheral_ and peripheral_schema_ must have the " + 
            "same length!")
    
    params["population"] = Placeholder(
        name=json_obj["placeholder_"]["name_"]
    )

    params["population"].thisptr = json_obj["placeholder_"]

    params["peripheral"] = [
        Placeholder(name=name) for name in json_obj["peripheral_"]
    ]

    params["population"].thisptr = _extract_schema(
        json_obj["population_schema_"],
        params["population"].thisptr
    )

    for i in range(len(json_obj["peripheral_schema_"])):
        params["peripheral"][i].thisptr = _extract_schema(
            json_obj["peripheral_schema_"][i],
            params["peripheral"][i].thisptr
        )

    return params


