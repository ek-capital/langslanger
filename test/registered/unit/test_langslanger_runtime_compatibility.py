import types
import unittest
from importlib.metadata import PackageNotFoundError

from sglang.srt.runtime_compatibility import (
    LANGSLANGER_CURRENT_BAND,
    OFFICIAL_SGLANG_IMAGE_BAND,
    select_cuda_runtime_band,
    validate_cuda_runtime_band,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLangSlangerRuntimeCompatibility(CustomTestCase):
    def test_selects_known_runtime_bands(self):
        self.assertEqual(
            select_cuda_runtime_band("2.11.0+cu130"),
            OFFICIAL_SGLANG_IMAGE_BAND,
        )
        self.assertEqual(
            select_cuda_runtime_band("2.13.0+cu130"),
            LANGSLANGER_CURRENT_BAND,
        )

    def test_strict_mode_rejects_official_image_band(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported CUDA runtime"):
            select_cuda_runtime_band("2.11.0+cu130", strict_current=True)

    def test_validates_official_flashinfer_band(self):
        versions = {
            "sglang-kernel": "0.4.5",
            "flashinfer_python": "0.6.14",
        }
        band = validate_cuda_runtime_band(
            "2.11.0+cu130",
            attention_backend="flashinfer",
            distribution_version=versions.__getitem__,
        )
        self.assertEqual(band, OFFICIAL_SGLANG_IMAGE_BAND)

    def test_rejects_cross_band_kernel_upgrade(self):
        versions = {
            "sglang-kernel": "0.4.6.post1",
            "flashinfer_python": "0.6.14",
        }
        with self.assertRaisesRegex(RuntimeError, "Do not upgrade one compiled"):
            validate_cuda_runtime_band(
                "2.11.0+cu130",
                attention_backend="flashinfer",
                distribution_version=versions.__getitem__,
            )

    def test_reports_missing_distribution(self):
        def missing(_package):
            raise PackageNotFoundError

        with self.assertRaisesRegex(RuntimeError, "sglang-kernel is not installed"):
            validate_cuda_runtime_band(
                "2.13.0+cu130",
                attention_backend="triton",
                distribution_version=missing,
            )


class TestMegaMoeRuntimeCapabilities(CustomTestCase):
    def test_in_place_weights_need_only_runtime_symbols(self):
        from sglang.srt.layers.moe.mega_moe_capabilities import (
            missing_sm90_fp8_mega_moe_symbols,
        )

        deep_gemm = types.SimpleNamespace(
            fp8_mega_moe=object(),
            get_symm_buffer_for_mega_moe=object(),
            mega_moe_pre_dispatch_sm90=object(),
        )
        self.assertEqual(
            missing_sm90_fp8_mega_moe_symbols(deep_gemm, use_in_place_weights=True),
            (),
        )

    def test_transformed_weights_report_missing_layout_symbols(self):
        from sglang.srt.layers.moe.mega_moe_capabilities import (
            missing_sm90_fp8_mega_moe_symbols,
        )

        deep_gemm = types.SimpleNamespace(
            fp8_mega_moe=object(),
            get_symm_buffer_for_mega_moe=object(),
            mega_moe_pre_dispatch_sm90=object(),
        )
        self.assertEqual(
            missing_sm90_fp8_mega_moe_symbols(deep_gemm, use_in_place_weights=False),
            (
                "transform_sf_into_required_layout",
                "transform_weights_for_mega_moe_sm90",
            ),
        )


if __name__ == "__main__":
    unittest.main()
