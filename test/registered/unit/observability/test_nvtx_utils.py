import unittest
from contextlib import nullcontext
from unittest.mock import patch

from sglang.srt.utils import nvtx_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNvtxUtils(unittest.TestCase):
    @patch.object(nvtx_utils.torch.autograd, "_profiler_enabled", return_value=False)
    @patch.object(nvtx_utils.torch.cuda, "is_available", return_value=True)
    @patch.object(nvtx_utils.torch.cuda.nvtx, "range", return_value=nullcontext())
    def test_explicit_range_falls_back_to_torch_nvtx(
        self, mock_range, _mock_cuda_available, _mock_profiler_enabled
    ):
        with patch.object(nvtx_utils, "_nvtx_module", None):
            with nvtx_utils.profile_range("model.attention.mla", nvtx_enabled=True):
                pass

        mock_range.assert_called_once_with("model.attention.mla")


if __name__ == "__main__":
    unittest.main()
