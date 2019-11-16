from getml.datasets import (
        make_numerical,
        make_categorical,
        make_discrete,
)

def test_make_numerical():
    population, peripheral = make_numerical(
            n_rows_population=10,
            n_rows_peripheral=40,
            random_state=2309)

    assert population.shape == (10, 4)
    assert peripheral.shape == (40, 3)
    assert population['targets'][0] == 1 

def test_make_categorical():
    population, peripheral = make_categorical(
            n_rows_population=10,
            n_rows_peripheral=50,
            random_state=2309)

    assert population.shape == (10, 4)
    assert peripheral.shape == (50, 3)
    assert population['targets'][0] == 4 

def test_make_discrete():
    population, peripheral = make_discrete(
            n_rows_population=10,
            n_rows_peripheral=50,
            random_state=2309)

    assert population.shape == (10, 4)
    assert peripheral.shape == (50, 3)
    assert population['targets'][0] == 1 

