import numpy as np

from gamegym import nested


def test_nested_flatten():
    assert all(nested.flatten([]) == [])
    assert all(nested.flatten({}) == [])
    assert all(nested.flatten([1.0]) == [1.0])
    assert all(nested.flatten([1.2], dtype=int) == [1])
    assert all(nested.flatten('ABC', dtype=object) == ['ABC'])
    assert all(nested.flatten([1, 2, (3.0, np.array([4, 5]), {'b': 8, 'a': (6, 7)})]) == [1, 2, 3, 4, 5, 6, 7, 8])


def test_nested_stack():
    d = [
        {'a': 1.0, 'b': (np.array([1,2,3]), np.float128(42), np.array([]), [0, 1, (2, )])},
        {'a': 2.0, 'b': (np.array([4,5,6]), np.float128(43), np.array([]), [1, 2, (3, )])},
        ]
    e = nested.stack(d)
    assert all(e['a'] == np.array([1.0, 2.0]))
    (b1, b2, b3, b4) = e['b']
    assert (b1 == np.array([[1,2,3], [4,5,6]])).all()
    assert all(b2 == np.array([42, 43]))
    assert b2.dtype == np.float128
    assert b3.shape == (2, 0)
    assert all(b4[0] == np.array([0, 1]))
    assert all(b4[1] == np.array([1, 2]))
    assert all(b4[2][0] == np.array([2, 3]))
