import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PB_PATH = os.path.join(ROOT, "tools", "plot_baselines.py")


class _Namespace(object):
    """Attribute view over a dict, so exec'd globals read like a module."""

    def __init__(self, d):
        self.__dict__ = d


# Loaded via exec (rather than an import statement) so this test module keeps
# the repo's "standard library plus pandas/numpy only" import surface.
_pb_globals = {"__file__": _PB_PATH, "__name__": "plot_baselines"}
with open(_PB_PATH) as _f:
    exec(compile(_f.read(), _PB_PATH, "exec"), _pb_globals)
pb = _Namespace(_pb_globals)


class TestPlotBaselinesDeterminism(unittest.TestCase):
    def test_svg_regenerates_byte_identical(self):
        first = pb.generate()
        second = pb.generate()
        self.assertEqual(first, second)

    def test_committed_svg_matches_current_generation(self):
        with open(pb.SVG_PATH) as f:
            on_disk = f.read()
        self.assertEqual(on_disk, pb.generate())


class TestPlotBaselinesValues(unittest.TestCase):
    def setUp(self):
        _, self.rows = pb.read_rows()
        with open(pb.SVG_PATH) as f:
            self.svg = f.read()

    def test_every_arm_plotted(self):
        for arm_id in pb.ARM_ORDER:
            self.assertIn('data-arm="%s"' % arm_id, self.svg)

    def test_plotted_marker_position_equals_csv(self):
        for arm_id in pb.ARM_ORDER:
            row = self.rows[arm_id]
            expected_x = pb.fx(row["accuracy"])
            expected_y = pb.fy(row["adjacent_substitution_rate"])

            circle = re.search(
                r'data-arm="%s"[^>]*cx="([\d.]+)"[^>]*cy="([\d.]+)"' % re.escape(arm_id),
                self.svg,
            )
            polygon = re.search(
                r'data-arm="%s"[^>]*points="([^"]+)"' % re.escape(arm_id),
                self.svg,
            )
            if circle:
                x, y = float(circle.group(1)), float(circle.group(2))
            else:
                self.assertIsNotNone(polygon, "no marker found for %s" % arm_id)
                pts = [
                    tuple(float(v) for v in pair.split(","))
                    for pair in polygon.group(1).split()
                ]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x = (min(xs) + max(xs)) / 2.0
                y = (min(ys) + max(ys)) / 2.0

            self.assertAlmostEqual(x, expected_x, places=2)
            self.assertAlmostEqual(y, expected_y, places=2)

    def test_ci_bars_span_csv_confidence_interval(self):
        for arm_id in pb.ARM_ORDER:
            row = self.rows[arm_id]
            x_lo = pb.fx(row["accuracy_ci_low"])
            x_hi = pb.fx(row["accuracy_ci_high"])
            self.assertIn("%.2f" % x_lo, self.svg)
            self.assertIn("%.2f" % x_hi, self.svg)

    def test_hatch_region_and_label_present(self):
        self.assertIn("noLmHatch", self.svg)
        self.assertIn("no language model scored yet", self.svg)


if __name__ == "__main__":
    unittest.main()
