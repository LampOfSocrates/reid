import tempfile
import unittest
from pathlib import Path

from PIL import Image

from coursework.src.utils.explore import (
    _read_index_line,
    _resolve_query_and_gallery,
    show_veri_good_and_junk,
)


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


class CourseworkExploreTests(unittest.TestCase):
    def test_read_index_line_extracts_integers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "gt_index_776.txt"
            index_file.write_text("1, 3\t5 -1\n2 4 6\n", encoding="utf-8")

            result = _read_index_line(index_file, 0)

            self.assertEqual(result, [1, 3, 5, -1])

    def test_resolve_query_and_gallery_returns_sorted_jpg_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_image(root / "image_query" / "0002.jpg", (255, 0, 0))
            _make_image(root / "image_query" / "0001.jpg", (0, 255, 0))
            _make_image(root / "image_test" / "0003.jpg", (0, 0, 255))
            _make_image(root / "image_test" / "0001.jpg", (255, 255, 0))

            query_files, gallery_files = _resolve_query_and_gallery(root)

            self.assertEqual([path.name for path in query_files], ["0001.jpg", "0002.jpg"])
            self.assertEqual([path.name for path in gallery_files], ["0001.jpg", "0003.jpg"])

    def test_show_veri_good_and_junk_returns_expected_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_image(root / "image_query" / "0001.jpg", (255, 0, 0))
            _make_image(root / "image_query" / "0002.jpg", (0, 255, 0))
            _make_image(root / "image_test" / "0001.jpg", (0, 0, 255))
            _make_image(root / "image_test" / "0002.jpg", (255, 255, 0))
            _make_image(root / "image_test" / "0003.jpg", (255, 0, 255))
            (root / "gt_index_776.txt").write_text("1 3\n2\n", encoding="utf-8")
            (root / "jk_index_776.txt").write_text("2\n3\n", encoding="utf-8")

            result = show_veri_good_and_junk(
                root=root,
                query_index=0,
                max_good=1,
                max_junk=1,
                show_plot=False,
            )

            self.assertEqual(Path(result["query_file"]).name, "0001.jpg")
            self.assertEqual([Path(path).name for path in result["good_files"]], ["0001.jpg"])
            self.assertEqual([Path(path).name for path in result["junk_files"]], ["0002.jpg"])
            self.assertEqual(result["num_good_total"], 2)
            self.assertEqual(result["num_junk_total"], 1)


if __name__ == "__main__":
    unittest.main()
