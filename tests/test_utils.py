from gamegym.utils import debug_assert, get_rng, uniform, np_uniform, sample_with_p, Distribution, cached
import numpy as np
import pytest
import random
import logging


def test_rng():
    get_rng(seed=42)
    get_rng(rng=np.random.RandomState(43))
    with pytest.raises(TypeError):
        get_rng(rng=random)
    with pytest.raises(TypeError):
        get_rng(rng=random.Random(41))


def test_debug_assert():
    debug_assert(lambda: True)
    with pytest.raises(AssertionError):
        debug_assert(lambda: False)


def test_uniform():
    assert uniform(1) == (1.0, )
    assert uniform(2) == (0.5, 0.5)
    assert len(uniform(10)) == 10
    assert sum(uniform(10)) == pytest.approx(1.0)
    assert (np_uniform(2) == np.array([0.5, 0.5])).all()
    assert len(np_uniform(10)) == 10
    assert np.sum(np_uniform(10)) == pytest.approx(1.0)


def test_sample_with_p():
    for i in range(50):
        assert sample_with_p("abcd", None)[0] in "abcd"
        assert sample_with_p("abcd", None)[1] == 0.25
        assert sample_with_p(8, None)[1] == 0.125
        assert sample_with_p(1, None)[1] == 1.0
        assert sample_with_p(None, [0.1, 0.8, 0.1])[0] in (0, 1, 2)
        assert sample_with_p(None, [0.1, 0.8, 0.1])[1] in (0.1, 0.8)
        assert sample_with_p(["a", "b"], [0.2, 0.8]) in (("a", 0.2), ("b", 0.8))


def test_distribution():
    for i in range(50):
        assert Distribution("abcd", None).sample() in "abcd"
        assert Distribution("abcd", None).sample_with_p()[1] == 0.25
        assert Distribution(8, None).sample_with_p()[1] == 0.125
        assert Distribution(1, None).sample_with_p()[1] == 1.0
        assert Distribution(None, [0.1, 0.8, 0.1]).sample_with_p()[0] in (0, 1, 2)
        assert Distribution(None, [0.1, 0.8, 0.1]).sample_with_p()[1] in (0.1, 0.8)
        assert Distribution(["a", "b"], [0.2, 0.8]).sample_with_p() in (("a", 0.2), ("b", 0.8))


class Foo:
    pass


def test_cached(tmpdir, caplog):
    runs = 0

    @cached
    def compute_x(a, b):
        nonlocal runs
        runs += 1
        return a + b

    @cached(prefix="asdf")
    def compute_y(c):
        nonlocal runs
        runs += 1
        return c

    with tmpdir.as_cwd():
        assert compute_x(1, 2) == 3
        assert runs == 1
        assert compute_x(1, 2) == 3
        assert runs == 1
        assert compute_x(1, b=2) == 3
        assert runs == 2
        assert compute_x(1, 3) == 4
        assert runs == 3
        assert compute_y(3) == 3
        assert runs == 4
        assert compute_y(3) == 3
        assert runs == 4
        foo = Foo()
        with caplog.at_level(logging.WARN):
            assert compute_y(foo) == foo
        assert 'contains a pointer address' in caplog.record_tuples[-1][2]
        assert runs == 5
