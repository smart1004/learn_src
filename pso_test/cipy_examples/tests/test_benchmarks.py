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
from examples import bench
from examples.gbest_pso import main as gbest
from examples.lbest_pso import main as lbest
from examples.pso_optimizer import main as pso_optimizer


@bench
def test_gbest_benchmark(benchmark):
    result = benchmark(gbest, 30, 1000)
    assert result is not None


@bench
def test_lbest_benchmark(benchmark):
    result = benchmark(lbest, 30, 1000)
    assert result is not None


@bench
def test_pso_benchmark(benchmark):
    result = benchmark(pso_optimizer, 30, 1000)
    assert result is not None