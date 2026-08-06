import unittest
from unittest.mock import patch

from sglang.srt.utils.profile_utils import (
    _start_cuda_profiler,
    _stop_cuda_profiler,
)


class TestCudaProfilerRankSynchronization(unittest.TestCase):
    def test_node_leader_starts_before_releasing_ranks(self):
        events = []
        with (
            patch(
                "sglang.srt.utils.profile_utils.torch.cuda.cudart"
            ) as mock_cudart,
            patch(
                "sglang.srt.utils.profile_utils.torch.distributed.barrier",
                side_effect=lambda group: events.append(("barrier", group)),
            ),
        ):
            mock_cudart.return_value.cudaProfilerStart.side_effect = lambda: (
                events.append(("start", None))
            )

            _start_cuda_profiler(first_rank_in_node=True, cpu_group="cpu-group")

        self.assertEqual(
            events,
            [("start", None), ("barrier", "cpu-group")],
        )

    def test_follower_waits_without_starting_profiler(self):
        events = []
        with (
            patch(
                "sglang.srt.utils.profile_utils.torch.cuda.cudart"
            ) as mock_cudart,
            patch(
                "sglang.srt.utils.profile_utils.torch.distributed.barrier",
                side_effect=lambda group: events.append(("barrier", group)),
            ),
        ):
            _start_cuda_profiler(first_rank_in_node=False, cpu_group="cpu-group")

        mock_cudart.assert_not_called()
        self.assertEqual(events, [("barrier", "cpu-group")])

    def test_node_leader_stops_only_after_all_ranks_arrive(self):
        events = []
        with (
            patch(
                "sglang.srt.utils.profile_utils.torch.cuda.cudart"
            ) as mock_cudart,
            patch(
                "sglang.srt.utils.profile_utils.torch.distributed.barrier",
                side_effect=lambda group: events.append(("barrier", group)),
            ),
        ):
            mock_cudart.return_value.cudaProfilerStop.side_effect = lambda: (
                events.append(("stop", None))
            )

            _stop_cuda_profiler(first_rank_in_node=True, cpu_group="cpu-group")

        self.assertEqual(
            events,
            [
                ("barrier", "cpu-group"),
                ("stop", None),
                ("barrier", "cpu-group"),
            ],
        )

    def test_follower_waits_on_both_sides_of_profiler_stop(self):
        events = []
        with (
            patch(
                "sglang.srt.utils.profile_utils.torch.cuda.cudart"
            ) as mock_cudart,
            patch(
                "sglang.srt.utils.profile_utils.torch.distributed.barrier",
                side_effect=lambda group: events.append(("barrier", group)),
            ),
        ):
            _stop_cuda_profiler(first_rank_in_node=False, cpu_group="cpu-group")

        mock_cudart.assert_not_called()
        self.assertEqual(
            events,
            [("barrier", "cpu-group"), ("barrier", "cpu-group")],
        )


if __name__ == "__main__":
    unittest.main()
