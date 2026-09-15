import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src_3d"))

from pn import ProportionalNavigation


class ProportionalNavigationTests(unittest.TestCase):
    def test_direction_uses_project_state_convention(self):
        direction = ProportionalNavigation.direction(0.0, 0.0)
        np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-12)

    def test_local_basis_is_orthonormal(self):
        forward, vertical, side = ProportionalNavigation.local_basis(
            np.array([2.0, -3.0, 4.0])
        )
        basis = np.vstack((forward, vertical, side))

        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(forward, side), vertical, atol=1e-12)

    def test_constant_los_returns_zero_command(self):
        guidance = ProportionalNavigation()
        missile = np.array([0.0, 0.0, 0.0, 300.0, 0.0, 0.0])
        target = np.array([1000.0, 0.0, 0.0, 100.0, 0.0, 0.0])

        np.testing.assert_allclose(guidance.solve(missile, target), np.zeros(2))

    def test_planar_los_rotation_commands_side_acceleration(self):
        guidance = ProportionalNavigation()
        missile = np.array([0.0, 0.0, 0.0, 300.0, 0.0, 0.0])
        target = np.array([1000.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        command = guidance.solve(missile, target)

        self.assertAlmostEqual(command[0], 0.0, places=12)
        self.assertGreater(command[1], 0.0)

    def test_command_respects_combined_acceleration_limit(self):
        guidance = ProportionalNavigation(navigation_constant=5.0, max_g=1.0)
        missile = np.array([0.0, 0.0, 0.0, 300.0, 0.0, 0.0])
        target = np.array([100.0, 100.0, 100.0, 1000.0, 0.0, np.pi])
        command = guidance.solve(missile, target)

        self.assertLessEqual(np.linalg.norm(command), 9.81 + 1e-12)


if __name__ == "__main__":
    unittest.main()
