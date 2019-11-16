"""
Generate samples of artificial data sets.
"""
import numpy as np
import pandas as pd

import getml.aggregations as aggregations

# -----------------------------------------------------------------------------

def _aggregate(table, aggregation, col, join_key):
    """Implements the aggregation."""

    if aggregation == aggregations.Avg:
        return table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).mean()
    
    elif aggregation == aggregations.Count: 
        return table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).count()

    elif aggregation == aggregations.CountDistinct: 
        series = table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        )[col].nunique()

        output = table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).count()

        output[col] = series 

        return output
    
    elif aggregation == aggregations.CountMinusCountDistinct: 
        series = table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        )[col].nunique()

        output = table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).count()

        output[col] -= series 

        return output
    
    elif aggregation == aggregations.Max: 
        return table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).max()
    
    elif aggregation == aggregations.Median: 
        return table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).median()
    
    elif aggregation == aggregations.Min: 
        return table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).min()
    
    elif aggregation == aggregations.Stddev: 
        return table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).std()
    
    elif aggregation == aggregations.Sum: 
        return table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).sum()
    
    elif aggregation == aggregations.Var: 
        return table[[col, join_key]].groupby(
            [join_key],
            as_index=False
        ).var()
    
    else:
        raise Exception("Aggregation '" + aggregation + "' not known!")


# -----------------------------------------------------------------------------

def make_categorical(n_rows_population=500, 
                     n_rows_peripheral=125000, 
                     random_state=None,
                     aggregation=aggregations.Count):
    """Generate a random dataset with categorical variables

    The dataset consists of a population table and one peripheral table.

    The peripheral table has 3 columns:

    * `column_01`: random categorical variable between '0' and '9' 
    * `join_key`: random integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1

    The population table has 4 columns:

    * `column_01`: random categorical variable between '0' and '9' 
    * `join_key`: unique integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1
    * `targets`: target variable. Defined as the number of matching entries in
      the peripheral table for which ``time_stamp_peripheral <
      time_stamp_population`` and the category in the peripheral table is not
      1, 2 or 9

    .. code-block:: sql

        SELECT aggregation( column_01/* )
        FROM POPULATION_TABLE t1
        LEFT JOIN PERIPHERAL_TABLE t2
        ON t1.join_key = t2.join_key
        WHERE (
           ( t2.column_01 != '1' AND t2.column_01 != '2' AND t2.column_01 != '9' )
        ) AND t2.time_stamps <= t1.time_stamps
        GROUP BY t1.join_key,
             t1.time_stamp;

    Args:
        n_rows_population (int, optional): Number of rows in the population table
        n_row_peripheral (int, optional): Number of rows in the peripheral table
        random_state (int, optional): Determines random number generation for
            dataset creation. Pass an int for reproducible output across
            multiple function calls.

   Returns:
       :class:`pandas.DataFrame`: Population table
       :class:`pandas.DataFrame`: Peripheral table

    """
    random = np.random.RandomState(random_state)

    population_table = pd.DataFrame()
    population_table["column_01"] = random.randint(0, 10, n_rows_population).astype(np.str)
    population_table["join_key"] = np.arange(n_rows_population)
    population_table["time_stamp_population"] = random.rand(n_rows_population)

    peripheral_table = pd.DataFrame()
    peripheral_table["column_01"] = random.randint(0, 10, n_rows_peripheral).astype(np.str)
    peripheral_table["join_key"] = random.randint(0, n_rows_population, n_rows_peripheral) 
    peripheral_table["time_stamp_peripheral"] = random.rand(n_rows_peripheral)

    # Compute targets
    temp = peripheral_table.merge(
        population_table[["join_key", "time_stamp_population"]],
        how="left",
        on="join_key"
    )

    # Apply some conditions
    temp = temp[
        (temp["time_stamp_peripheral"] <= temp["time_stamp_population"]) &
        (temp["column_01"] != "1") &
        (temp["column_01"] != "2") &
        (temp["column_01"] != "9")
    ]

    # Define the aggregation
    temp = _aggregate(temp, aggregation, "column_01", "join_key")

    temp = temp.rename(index=str, columns={"column_01": "targets"})

    population_table = population_table.merge(
        temp,
        how="left",
        on="join_key"
    )

    del temp

    population_table = population_table.rename(
        index=str, columns={"time_stamp_population": "time_stamp"})

    peripheral_table = peripheral_table.rename(
        index=str, columns={"time_stamp_peripheral": "time_stamp"})

    # Replace NaN targets with 0.0 - target values may never be NaN!.
    population_table.targets = np.where(
            np.isnan(population_table['targets']), 
            0, 
            population_table['targets'])

    return population_table, peripheral_table


# -----------------------------------------------------------------------------

def make_discrete(n_rows_population=500, 
                  n_rows_peripheral=125000, 
                  random_state=None,
                  aggregation=aggregations.Count):
    """Generate a random dataset with categorical variables

    The dataset consists of a population table and one peripheral table.

    The peripheral table has 3 columns:

    * `column_01`: random integer between -10 and 10
    * `join_key`: random integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1

    The population table has 4 columns:

    * `column_01`: random number between -1 and 1
    * `join_key`: unique integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1
    * `targets`: target variable. Defined as the minimum value greater than 0
       in the peripheral table for which ``time_stamp_periperhal <
       time_stamp_population`` and the join key machtes

    .. code-block:: sql

        SELECT aggregation( column_01/* )
        FROM POPULATION t1
        LEFT JOIN PERIPHERAL t2
        ON t1.join_key = t2.join_key
        WHERE (
           ( t2.column_01 > 0 )
        ) AND t2.time_stamp <= t1.time_stamp
        GROUP BY t1.join_key,
                 t1.time_stamp;

    Args:
        n_rows_population (int, optional): Number of rows in the population table
        n_row_peripheral (int, optional): Number of rows in the peripheral table
        random_state (int, optional): Determines random number generation for
            dataset creation. Pass an int for reproducible output across
            multiple function calls.

   Returns:
       :class:`pandas.DataFrame`: Population table
       :class:`pandas.DataFrame`: Peripheral table

    """
    random = np.random.RandomState(random_state)

    population_table = pd.DataFrame()
    population_table["column_01"] = random.randint(0, 10, n_rows_population).astype(np.str)
    population_table["join_key"] = np.arange(n_rows_population)
    population_table["time_stamp_population"] = random.rand(n_rows_population)

    peripheral_table = pd.DataFrame()
    peripheral_table["column_01"] = random.randint(-11, 11, n_rows_peripheral)
    peripheral_table["join_key"] = random.randint(0, n_rows_population, n_rows_peripheral) 
    peripheral_table["time_stamp_peripheral"] = random.rand(n_rows_peripheral)

    # Compute targets
    temp = peripheral_table.merge(
        population_table[["join_key", "time_stamp_population"]],
        how="left",
        on="join_key"
    )

    # Apply some conditions
    temp = temp[
        (temp["time_stamp_peripheral"] <= temp["time_stamp_population"]) &
        (temp["column_01"] > 0.0)
    ]

    # Define the aggregation
    temp = _aggregate(temp, aggregation, "column_01", "join_key")

    temp = temp.rename(index=str, columns={"column_01": "targets"})

    population_table = population_table.merge(
        temp,
        how="left",
        on="join_key"
    )

    del temp

    population_table = population_table.rename(
        index=str, columns={"time_stamp_population": "time_stamp"})

    peripheral_table = peripheral_table.rename(
        index=str, columns={"time_stamp_peripheral": "time_stamp"})

    # Replace NaN targets with 0.0 - target values may never be NaN!.
    population_table.targets = np.where(
            np.isnan(population_table['targets']), 
            0, 
            population_table['targets'])

    return population_table, peripheral_table

# -----------------------------------------------------------------------------

def make_numerical(n_rows_population=500, 
                   n_rows_peripheral=125000, 
                   random_state=None,
                   aggregation=aggregations.Count):
    """Generate a random dataset with continous numerical variables

    The dataset consists of a population table and one peripheral table.

    The peripheral table has 3 columns:

    * `column_01`:  random number between -1 and 1
    * `join_key`: random integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1

    The population table has 4 columns:

    * `column_01`:  random number between -1 and 1
    * `join_key`: unique integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1
    * `targets`: target variable. Defined as the number of matching entries in
      the peripheral table for which ``time_stamp_periperhal <
      time_stamp_population < time_stamp_peripheral + 0.5``

    .. code-block:: sql

        SELECT aggregation( column_01/* )
        FROM POPULATION t1
        LEFT JOIN PERIPHERAL t2
        ON t1.join_key = t2.join_key
        WHERE (
           ( t1.time_stamp - t2.time_stamp <= 0.5 )
        ) AND t2.time_stamp <= t1.time_stamp
        GROUP BY t1.join_key,
             t1.time_stamp;

    Args:
        n_rows_population (int, optional): Number of rows in the population table
        n_row_peripheral (int, optional): Number of rows in the peripheral table
        random_state (int, optional): Determines random number generation for
            dataset creation. Pass an int for reproducible output across
            multiple function calls.

   Returns:
       :class:`pandas.DataFrame`: Population table
       :class:`pandas.DataFrame`: Peripheral table

    """
    random = np.random.RandomState(random_state)

    population_table = pd.DataFrame()
    population_table["column_01"] = random.rand(n_rows_population) * 2.0 - 1.0
    population_table["join_key"] = np.arange(n_rows_population)
    population_table["time_stamp_population"] = random.rand(n_rows_population)

    peripheral_table = pd.DataFrame()
    peripheral_table["column_01"] = random.rand(n_rows_peripheral) * 2.0 - 1.0
    peripheral_table["join_key"] = random.randint(0, n_rows_population, n_rows_peripheral) 
    peripheral_table["time_stamp_peripheral"] = random.rand(n_rows_peripheral)

    # Compute targets
    temp = peripheral_table.merge(
        population_table[["join_key", "time_stamp_population"]],
        how="left",
        on="join_key"
    )

    # Apply some conditions
    temp = temp[
        (temp["time_stamp_peripheral"] <= temp["time_stamp_population"]) &
        (temp["time_stamp_peripheral"] >= temp["time_stamp_population"] - 0.5)
    ]

    # Define the aggregation
    temp = _aggregate(temp, aggregation, "column_01", "join_key")

    temp = temp.rename(index=str, columns={"column_01": "targets"})

    population_table = population_table.merge(
        temp,
        how="left",
        on="join_key"
    )

    del temp

    population_table = population_table.rename(
        index=str, columns={"time_stamp_population": "time_stamp"})

    peripheral_table = peripheral_table.rename(
        index=str, columns={"time_stamp_peripheral": "time_stamp"})

    # Replace NaN targets with 0.0 - target values may never be NaN!.
    population_table.targets = np.where(
            np.isnan(population_table['targets']), 
            0, 
            population_table['targets'])

    return population_table, peripheral_table

# -----------------------------------------------------------------------------

def make_same_units_categorical(n_rows_population=500, 
                                n_rows_peripheral=125000, 
                                random_state=None,
                                aggregation=aggregations.Count):
    """Generate a random dataset with categorical variables

    The dataset consists of a population table and one peripheral table.

    The peripheral table has 3 columns:

    * `column_01`: random categorical variable between '0' and '9' 
    * `join_key`: random integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1

    The population table has 4 columns:

    * `column_01`: random categorical variable between '0' and '9' 
    * `join_key`: unique integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1
    * `targets`: target variable. Defined as the number of matching entries in
      the peripheral table for which ``time_stamp_peripheral <
      time_stamp_population`` and the category in the peripheral table is not
      1, 2 or 9

    .. code-block:: sql

    SELECT aggregation( column_01/* )
    FROM POPULATION_TABLE t1
    LEFT JOIN PERIPHERAL_TABLE t2
    ON t1.join_key = t2.join_key
    WHERE (
       ( t1.column_01 == t2.column_01 )
    ) AND t2.time_stamps <= t1.time_stamps
    GROUP BY t1.join_key,
         t1.time_stamp;

    Args:
        n_rows_population (int, optional): Number of rows in the population table
        n_row_peripheral (int, optional): Number of rows in the peripheral table
        random_state (int, optional): Determines random number generation for
            dataset creation. Pass an int for reproducible output across
            multiple function calls.

   Returns:
       :class:`pandas.DataFrame`: Population table
       :class:`pandas.DataFrame`: Peripheral table

    """
    population_table = pd.DataFrame()
    population_table["column_01_population"] = (np.random.rand(
        n_rows_population)*10.0).astype(np.int).astype(np.str)
    population_table["join_key"] = range(n_rows_population)
    population_table["time_stamp_population"] = np.random.rand(n_rows_population)

    peripheral_table = pd.DataFrame()
    peripheral_table["column_01_peripheral"] = (np.random.rand(
        n_rows_peripheral)*10.0).astype(np.int).astype(np.str)
    peripheral_table["column_02"] = np.random.rand(n_rows_peripheral) * 2.0 - 1.0
    peripheral_table["join_key"] = [
        int(float(n_rows_population) * np.random.rand(1)[0]) for i in range(n_rows_peripheral)]
    peripheral_table["time_stamp_peripheral"] = np.random.rand(n_rows_peripheral)

    # ----------------

    temp = peripheral_table.merge(
        population_table[["join_key", "time_stamp_population",
        "column_01_population"]],
        how="left",
        on="join_key"
    )

    # Apply some conditions
    temp = temp[
        (temp["time_stamp_peripheral"] <= temp["time_stamp_population"]) &
        (temp["column_01_peripheral"] == temp["column_01_population"])
    ]

    # Define the aggregation
    temp = _aggregate(temp, aggregation, "column_02", "join_key")

    temp = temp.rename(index=str, columns={"column_02": "targets"})

    population_table = population_table.merge(
        temp,
        how="left",
        on="join_key"
    )

    population_table = population_table.rename(
      index=str, 
      columns={"column_01_population": "column_01"}
    )

    peripheral_table = peripheral_table.rename(
      index=str, 
      columns={"column_01_peripheral": "column_01"}
    )

    del temp

    # ----------------

    population_table = population_table.rename(
        index=str, columns={"time_stamp_population": "time_stamp"})

    peripheral_table = peripheral_table.rename(
        index=str, columns={"time_stamp_peripheral": "time_stamp"})

    # ----------------

    # Replace NaN targets with 0.0 - target values may never be NaN!.
    population_table["targets"] = [
        0.0 if val != val else val for val in population_table["targets"]
    ]
    
    # ----------------

    return population_table, peripheral_table

# -----------------------------------------------------------------------------

def make_same_units_numerical(n_rows_population=500, 
                   n_rows_peripheral=125000, 
                   random_state=None,
                   aggregation=aggregations.Count):
    """Generate a random dataset with continous numerical variables

    The dataset consists of a population table and one peripheral table.

    The peripheral table has 3 columns:

    * `column_01`:  random number between -1 and 1
    * `join_key`: random integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1

    The population table has 4 columns:

    * `column_01`:  random number between -1 and 1
    * `join_key`: unique integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1
    * `targets`: target variable. Defined as the number of matching entries in
      the peripheral table for which ``time_stamp_periperhal <
      time_stamp_population < time_stamp_peripheral + 0.5``

    .. code-block:: sql

        SELECT aggregation( column_01/* )
        FROM POPULATION t1
        LEFT JOIN PERIPHERAL t2
        ON t1.join_key = t2.join_key
        WHERE (
           ( t1.column_01 - t2.column_01 <= 0.5 )
        ) AND t2.time_stamp <= t1.time_stamp
        GROUP BY t1.join_key,
             t1.time_stamp;

    Args:
        n_rows_population (int, optional): Number of rows in the population table
        n_row_peripheral (int, optional): Number of rows in the peripheral table
        random_state (int, optional): Determines random number generation for
            dataset creation. Pass an int for reproducible output across
            multiple function calls.

   Returns:
       :class:`pandas.DataFrame`: Population table
       :class:`pandas.DataFrame`: Peripheral table

    """
    random = np.random.RandomState(random_state)

    population_table = pd.DataFrame()
    population_table["column_01_population"] = np.random.rand(n_rows_population) * 2.0 - 1.0
    population_table["join_key"] = range(n_rows_population)
    population_table["time_stamp_population"] = np.random.rand(n_rows_population)

    peripheral_table = pd.DataFrame()
    peripheral_table["column_01_peripheral"] = np.random.rand(n_rows_peripheral) * 2.0 - 1.0
    peripheral_table["join_key"] = [
        int(float(n_rows_population) * np.random.rand(1)[0]) for i in range(n_rows_peripheral)]
    peripheral_table["time_stamp_peripheral"] = np.random.rand(n_rows_peripheral)

    # ----------------

    temp = peripheral_table.merge(
        population_table[["join_key", "time_stamp_population",
        "column_01_population"]],
        how="left",
        on="join_key"
    )

    # Apply some conditions
    temp = temp[
        (temp["time_stamp_peripheral"] <= temp["time_stamp_population"]) &
        (temp["column_01_peripheral"] > temp["column_01_population"] - 0.5)
    ]

    # Define the aggregation
    temp = temp[["column_01_peripheral", "join_key"]].groupby(
        ["join_key"],
        as_index=False
    ).count()

    temp = temp.rename(index=str, columns={"column_01_peripheral": "targets"})

    population_table = population_table.merge(
        temp,
        how="left",
        on="join_key"
    )

    population_table = population_table.rename(
      index=str, 
      columns={"column_01_population": "column_01"}
    )

    peripheral_table = peripheral_table.rename(
      index=str, 
      columns={"column_01_peripheral": "column_01"}
    )

    del temp

    # ----------------

    population_table = population_table.rename(
        index=str, columns={"time_stamp_population": "time_stamp"})

    peripheral_table = peripheral_table.rename(
        index=str, columns={"time_stamp_peripheral": "time_stamp"})

    # ----------------

    # Replace NaN targets with 0.0 - target values may never be NaN!.
    population_table["targets"] = [
        0.0 if val != val else val for val in population_table["targets"]
    ]

    return population_table, peripheral_table

# -----------------------------------------------------------------------------

def make_snowflake(n_rows_population=500, 
                   n_rows_peripheral1=5000, 
                   n_rows_peripheral2=125000, 
                   aggregation1=aggregations.Sum,
                   aggregation2=aggregations.Count,
                   random_state=None):
    """Generate a random dataset with continous numerical variables

    The dataset consists of a population table and two peripheral tables.

    The first peripheral table has 4 columns:

    * `column_01`:  random number between -1 and 1
    * `join_key`: random integer in the range from 0 to ``n_rows_population``
    * `join_key2`: unique integer in the range from 0 to ``n_rows_peripheral1``
    * `time_stamp`: random number between 0 and 1

    The second peripheral table has 3 columns:

    * `column_01`:  random number between -1 and 1
    * `join_key2`: random integer in the range from 0 to ``n_rows_peripheral1``
    * `time_stamp`: random number between 0 and 1
    
    The population table has 4 columns:

    * `column_01`:  random number between -1 and 1
    * `join_key`: unique integer in the range from 0 to ``n_rows_population``
    * `time_stamp`: random number between 0 and 1
    * `targets`: target variable as defined by the SQL block below:

    .. code-block:: sql

        SELECT aggregation1( feature_1_1/* )
        FROM POPULATION t1
        LEFT JOIN (
            SELECT aggregation2( t4.column_01 ) AS feature_1_1
            FROM PERIPHERAL t3
            LEFT JOIN PERIPHERAL2 t4
            ON t3.join_key2 = t4.join_key2
            WHERE (
               ( t3.time_stamp - t4.time_stamp <= 0.5 )
            ) AND t4.time_stamp <= t3.time_stamp
            GROUP BY t3.join_key,
                 t3.time_stamp
        ) t2
        ON t1.join_key = t2.join_key
        WHERE t2.time_stamp <= t1.time_stamp
        GROUP BY t1.join_key,
             t1.time_stamp;

    Args:
        n_rows_population (int, optional): Number of rows in the population table
        n_row_peripheral1 (int, optional): Number of rows in the first peripheral table
        n_row_peripheral2 (int, optional): Number of rows in the second peripheral table
        aggregation1 (string, optional): Aggregation to be used to aggregate the first peripheral table
        aggregation2 (string, optional): Aggregation to be used to aggregate the second peripheral table
        random_state (int, optional): Determines random number generation for
            dataset creation. Pass an int for reproducible output across
            multiple function calls.

   Returns:
       :class:`pandas.DataFrame`: Population table
       :class:`pandas.DataFrame`: First peripheral table
       :class:`pandas.DataFrame`: Second peripheral table

    """
    random = np.random.RandomState(random_state)
    
    population_table = pd.DataFrame()
    population_table["column_01"] = np.random.rand(n_rows_population) * 2.0 - 1.0
    population_table["join_key"] = range(n_rows_population)
    population_table["time_stamp_population"] = np.random.rand(n_rows_population)

    peripheral_table = pd.DataFrame()
    peripheral_table["column_01"] = np.random.rand(n_rows_peripheral1) * 2.0 - 1.0
    peripheral_table["join_key"] = [
        int(float(n_rows_population) * np.random.rand(1)[0]) for i in range(n_rows_peripheral1)]
    peripheral_table["join_key2"] = range(n_rows_peripheral1)
    peripheral_table["time_stamp_peripheral"] = np.random.rand(n_rows_peripheral1)

    peripheral_table2 = pd.DataFrame()
    peripheral_table2["column_01"] = np.random.rand(n_rows_peripheral2) * 2.0 - 1.0
    peripheral_table2["join_key2"] = [
        int(float(n_rows_peripheral1) * np.random.rand(1)[0]) for i in range(n_rows_peripheral2)]
    peripheral_table2["time_stamp_peripheral2"] = np.random.rand(n_rows_peripheral2)

    # ----------------
    # Merge peripheral_table with peripheral_table2

    temp = peripheral_table2.merge(
        peripheral_table[["join_key2", "time_stamp_peripheral"]],
        how="left",
        on="join_key2"
    )

    # Apply some conditions
    temp = temp[
        (temp["time_stamp_peripheral2"] <= temp["time_stamp_peripheral"]) &
        (temp["time_stamp_peripheral2"] >= temp["time_stamp_peripheral"] - 0.5)
    ]

    # Define the aggregation
    temp = _aggregate(temp, aggregation2, "column_01", "join_key2")

    temp = temp.rename(index=str, columns={"column_01": "temporary"})

    peripheral_table = peripheral_table.merge(
        temp,
        how="left",
        on="join_key2"
    )

    del temp

    # Replace NaN with 0.0
    peripheral_table["temporary"] = [
        0.0 if val != val else val for val in peripheral_table["temporary"]
    ]

    # ----------------
    # Merge population_table with peripheral_table

    temp2 = peripheral_table.merge(
        population_table[["join_key", "time_stamp_population"]],
        how="left",
        on="join_key"
    )

    # Apply some conditions
    temp2 = temp2[
        (temp2["time_stamp_peripheral"] <= temp2["time_stamp_population"])
    ]

    # Define the aggregation
    temp2 = _aggregate(temp2, aggregation1, "temporary", "join_key")
    
    temp2 = temp2.rename(index=str, columns={"temporary": "targets"})

    population_table = population_table.merge(
        temp2,
        how="left",
        on="join_key"
    )

    del temp2

    # Replace NaN targets with 0.0 - target values may never be NaN!.
    population_table["targets"] = [
        0.0 if val != val else val for val in population_table["targets"]
    ]

    # Remove temporary column.
    del peripheral_table["temporary"]

    # ----------------

    population_table = population_table.rename(
        index=str, columns={"time_stamp_population": "time_stamp"})

    peripheral_table = peripheral_table.rename(
        index=str, columns={"time_stamp_peripheral": "time_stamp"})

    peripheral_table2 = peripheral_table2.rename(
        index=str, columns={"time_stamp_peripheral2": "time_stamp"})

    # ----------------

    return population_table, peripheral_table, peripheral_table2

# -----------------------------------------------------------------------------


