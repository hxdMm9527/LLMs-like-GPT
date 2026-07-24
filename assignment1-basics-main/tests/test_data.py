import math
from collections import Counter

import numpy as np
import pytest

from .adapters import run_get_batch


def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # 合理性检查，确保随机样本确实有一定随机性。
    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # 确保形状正确
        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # 确保 y 总是偏移 1
        # Make sure the y's are always offset by 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # 确保我们永远不会采样到无效的起始索引
    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # 看到每个起始索引的期望次数
    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices))
    )
    # 预期结果的范围 (mu +/- 5sigma)。对于给定索引，这应该在 99.99994% 的情况下成立。
    # 因此，在 93 个可能的起始索引的情况下，整个测试应该有 99.9944202% 的时间通过
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # 我们假设 cuda:99 是一个无效的设备序号。
        # 在此添加此测试以确保 device 参数被正确处理。
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)
