from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_abide_connectomes import EXPECTED_REGIONS, fisher_z_connectome, read_aal_timeseries


class ConnectomeBuilderTests(unittest.TestCase):
    def test_parser_preserves_header_and_numeric_shape(self) -> None:
        labels = [f"#{index}" for index in range(EXPECTED_REGIONS)]
        rows = [" ".join(str(float(index + offset)) for index in range(EXPECTED_REGIONS)) for offset in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.1D"
            path.write_text("\t".join(labels) + "\n" + "\n".join(rows) + "\n")
            actual_labels, values = read_aal_timeseries(path)
        self.assertEqual(actual_labels, tuple(labels))
        self.assertEqual(values.shape, (3, EXPECTED_REGIONS))

    def test_fisher_z_is_symmetric_and_zero_diagonal(self) -> None:
        generator = np.random.default_rng(4)
        timeseries = generator.normal(size=(100, EXPECTED_REGIONS))
        connectome, clipped = fisher_z_connectome(timeseries)
        self.assertEqual(connectome.shape, (EXPECTED_REGIONS, EXPECTED_REGIONS))
        self.assertTrue(np.allclose(connectome, connectome.T))
        self.assertTrue(np.allclose(np.diag(connectome), 0.0))
        self.assertEqual(clipped, 0)

    def test_zero_variance_roi_is_rejected(self) -> None:
        labels = [f"#{index}" for index in range(EXPECTED_REGIONS)]
        values = np.ones((3, EXPECTED_REGIONS))
        values[:, 1:] += np.arange(3)[:, None]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.1D"
            lines = ["\t".join(labels)] + [" ".join(map(str, row)) for row in values]
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "zero-variance"):
                read_aal_timeseries(path)


if __name__ == "__main__":
    unittest.main()
