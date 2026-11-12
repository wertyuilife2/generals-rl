"""CUDA validation remains effective without synchronizing every valid action."""
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize('invalid', ['out_of_range', 'masked'])
def test_invalid_cuda_action_still_fails_in_an_isolated_context(invalid):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    program = '''
import torch
from generals_rl.policy.actions import EncodedActionSpace
space = EncodedActionSpace(3, 3)
mask = torch.zeros(1, space.size, dtype=torch.bool, device='cuda')
mask[:, 0] = True
space.validate(torch.tensor([ACTION], device='cuda'), mask)
torch.cuda.synchronize()
'''.replace('ACTION', '1000' if invalid == 'out_of_range' else '1')
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'device-side assert' in result.stderr or 'out of range or masked' in result.stderr
