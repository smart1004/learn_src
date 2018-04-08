# Copyright 2016 Andrich van Wyk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Tests ensuring examples execute
"""
import pytest

from examples import example
from examples.gbest_pso import main as gbest
from examples.gc_pso import main as gc
from examples.lbest_pso import main as lbest
from examples.pso_optimizer import main as pso_optimizer


@pytest.mark.parametrize("dimension", [
    1,
    30
])
@pytest.mark.parametrize("iterations", [
    3
])
@example
def test_gbest_pso(dimension, iterations):
    gbest(dimension, iterations)


@pytest.mark.parametrize("dimension", [
    1,
    30
])
@pytest.mark.parametrize("iterations", [
    3
])
@example
def test_lbest_pso(dimension, iterations):
    lbest(dimension, iterations)


@pytest.mark.parametrize("dimension", [
    1,
    30
])
@pytest.mark.parametrize("iterations", [
    3
])
@example
def test_gc_pso(dimension, iterations):
    gc(dimension, iterations)


@pytest.mark.parametrize("dimension", [
    1,
    30
])
@pytest.mark.parametrize("iterations", [
    3
])
@example
def test_optimizer(dimension, iterations):
    pso_optimizer(dimension, iterations)
