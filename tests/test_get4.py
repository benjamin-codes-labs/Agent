import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from sentinel import GET4


@pytest.mark.parametrize("name,expected", [
    ("img_cell001_BSE.tif", ("cell001", "bse")),
    ("img_cell_002_Inlens.TIFF", ("cell_002", "inlens")),
    ("img_BSE_cell_ETD.png", ("BSE_cell", "etd")),
])
def test_parse_batch_image_name(name, expected):
    assert GET4.parse_image_identity(name) == expected


@pytest.mark.parametrize("name", ["cell_BSE.tif", "img__BSE.tif", "img_cell_SE.tif", "img_cell_BSE_extra.tif"])
def test_reject_invalid_batch_image_name(name):
    with pytest.raises(ValueError, match="img_batteryname_filter"):
        GET4.parse_image_identity(name)


def _batch_folders(root):
    for batch in ("Batch_1", "Batch_2", "Batch_3"):
        folder = root / batch
        folder.mkdir()
        for name in ("img_cell001_BSE.tif", "img_cell002_BSE.tif", "img_cell001_Inlens.tif"):
            (folder / name).touch()
    return root


def test_group_by_folder_not_battery_name(tmp_path):
    _batch_folders(tmp_path)
    grouped = GET4.collect_batch_groups([tmp_path])
    assert list(grouped) == ["Batch_1", "Batch_2", "Batch_3"]
    assert set(grouped["Batch_1"]) == {"bse", "inlens"}
    assert len(grouped["Batch_1"]["bse"]) == 2
    assert all(path.parent.name == "Batch_1" for path in grouped["Batch_1"]["bse"])


def test_explicit_folders_and_detector_selection(tmp_path):
    _batch_folders(tmp_path)
    folders = [tmp_path / "Batch_1", tmp_path / "Batch_2"]
    grouped = GET4.collect_batch_groups(folders + [folders[0]], "InLens")
    assert list(grouped) == ["Batch_1", "Batch_2"]
    assert all(set(filters) == {"inlens"} for filters in grouped.values())


def test_reject_duplicate_battery_detector_images(tmp_path):
    _batch_folders(tmp_path)
    (tmp_path / "Batch_1" / "img_cell001_bse.png").touch()
    with pytest.raises(ValueError, match="duplicate"):
        GET4.collect_batch_groups([tmp_path])


def test_reject_ambiguous_batch_folder_names(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
        _batch_folders(root)
    with pytest.raises(ValueError, match="same batch name"):
        GET4.collect_batch_groups([root / "Batch_1" for root in roots])


def _phase(phi, n=3):
    return {
        "phi": phi, "se_ci": 0.01, "tau2": 0.0001,
        "tau2_source": "estimated" if n > 1 else "NOT ESTIMABLE from one image",
        "n_images": n, "variant_shifts": [0.01, -0.01, 0.0, 0.005, 0.002],
        "systematic_range": [phi - 0.01, phi + 0.01], "systematic_halfwidth": 0.01,
    }


def _summary(phi, detector="bse", n=3):
    return {"detector": detector, "target_pixel_nm": 10.0, "images": [],
            "phases": {"pore": _phase(phi, n)}}


def test_compare_summaries_preserves_existing_difference_calculation():
    a, b = _summary(0.30), _summary(0.27)
    result = GET4.compare_batch_summaries(a, b, "Batch_1", "Batch_2", {"pore": 0.02}, 0.10)
    row = result["phases"][0]
    assert row["delta"] == pytest.approx(-0.03)
    assert row["delta_rel"] == pytest.approx(-0.10)
    assert row["tolerance"] == 0.02
    assert row["interval95"][0] < row["delta"] < row["interval95"][1]
    assert row["systematic_kind"] == "common-mode cancelled"


def test_compare_summaries_rejects_cross_detector_comparison():
    with pytest.raises(ValueError, match="different detectors"):
        GET4.compare_batch_summaries(_summary(0.3), _summary(0.2, "etd"), "a", "b", {}, 0.1)


def test_cross_batch_imaging_difference_disables_bias_cancellation():
    a, b = _summary(0.3), _summary(0.2)
    a["images"] = [{"image": "a.tif", "quality": {"sharpness": 1.0}, "imaging_flags": []}]
    b["images"] = [{"image": "b.tif", "quality": {"sharpness": 2.0}, "imaging_flags": []}]
    result = GET4.compare_batch_summaries(a, b, "a", "b", {}, 0.1)
    assert result["flagged_images"]
    assert result["phases"][0]["systematic_kind"] == "independent (imaging differs or old JSON)"


def test_single_image_caveat_survives_comparison():
    a, b = _summary(0.3, n=1), _summary(0.2, n=1)
    result = GET4.compare_batch_summaries(a, b, "a", "b", {}, 0.1)
    assert any("variation unknown" in caveat for caveat in result["phases"][0]["caveats"])


def test_legacy_compare_json_files_still_works(tmp_path):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, phi in zip(paths, (0.3, 0.2)):
        path.write_text(json.dumps(_summary(phi)))
    result = GET4.compare_lots(*paths, {}, 0.1)
    assert result["phases"][0]["delta"] == pytest.approx(-0.1)
    assert result["lot_a"] == str(paths[0])


def _image_folders(root):
    rng = np.random.default_rng(23)
    for i in range(1, 4):
        folder = root / f"Batch_{i}"
        folder.mkdir()
        for detector in ("BSE", "ETD", "Inlens"):
            for j in range(2):
                cells = rng.choice([25, 125, 225], size=(12, 12), p=[0.2 + i * 0.04, 0.5 - i * 0.04, 0.3])
                image = np.repeat(np.repeat(cells, 8, axis=0), 8, axis=1)
                image = np.clip(image + rng.normal(0, 1, image.shape), 0, 255).astype(np.uint8)
                tifffile.imwrite(folder / f"img_cell{j}_{detector}.tif", image)


def _args(root):
    return argparse.Namespace(
        inputs=[str(root)], detector="all", out=str(root / "output"),
        tol="pore=0.02", tol_rel=0.1, baseline=None,
        page=0, px_size=10.0, target_px=None, bin=None,
        crop_bottom=0, tilt_deg=None, max_lag=8,
        no_flatten=True, no_destripe=True, no_plots=True,
    )


def test_all_batch_comparisons_end_to_end(tmp_path):
    _image_folders(tmp_path)
    args = _args(tmp_path)
    args.no_plots = False
    report = GET4.compare_all_batches(args)
    assert report["batches"] == ["Batch_1", "Batch_2", "Batch_3"]
    assert set(report["detectors"]) == {"bse", "etd", "inlens"}
    for detector, data in report["detectors"].items():
        assert data["target_pixel_nm"] == 10.0
        assert len(data["comparisons"]) == 3
        assert {(c["lot_a"], c["lot_b"]) for c in data["comparisons"]} == {
            ("Batch_1", "Batch_2"), ("Batch_1", "Batch_3"), ("Batch_2", "Batch_3")}
        for name, summary in data["batches"].items():
            assert summary["detector"] == detector
            assert summary["batch"] == name
            assert all(phase["n_images"] == 2 for phase in summary["phases"].values())
            assert not (tmp_path / "output" / detector / name).exists()
        if detector != "bse":
            assert any("intensity" in caveat for caveat in data["caveats"])
    saved = json.loads((tmp_path / "output" / "all_batch_comparisons.json").read_text())
    assert len(saved["detectors"]["bse"]["comparisons"]) == 3
    assert [path.name for path in (tmp_path / "output").iterdir()] == ["all_batch_comparisons.json"]
    assert any("not adjusted" in caveat for caveat in saved["caveats"])


def test_missing_detector_does_not_mix_groups(tmp_path, monkeypatch):
    _batch_folders(tmp_path)
    (tmp_path / "Batch_1" / "img_extra_ETD.tif").touch()
    args = _args(tmp_path)
    monkeypatch.setattr(GET4, "read_meta", lambda *a: {"pixel_size_nm": 10.0})
    calls = []

    def analyse(paths, metas, target, note, args, baseline=None, batch_name=None, detector=None, write_reports=True):
        assert not write_reports
        calls.append((batch_name, detector))
        return {"batch": batch_name, **_summary(0.3, detector)}

    monkeypatch.setattr(GET4, "analyse_batch_files", analyse)
    result = GET4.compare_all_batches(args)
    etd = result["detectors"]["etd"]
    assert etd["comparisons"] == []
    assert etd["missing_batches"] == ["Batch_2", "Batch_3"]
    assert ("Batch_2", "etd") not in calls
    assert len(result["detectors"]["bse"]["comparisons"]) == 3


def test_common_scale_uses_all_batches_in_a_detector(tmp_path, monkeypatch):
    _batch_folders(tmp_path)
    args = _args(tmp_path)
    args.detector, args.px_size = "bse", None
    monkeypatch.setattr(GET4, "read_meta", lambda path, page: {
        "pixel_size_nm": {"Batch_1": 5.0, "Batch_2": 10.0, "Batch_3": 20.0}[path.parent.name]})
    targets = []

    def analyse(paths, metas, target, note, args, **kwargs):
        targets.append(target)
        return {**_summary(0.3), "target_pixel_nm": target}

    monkeypatch.setattr(GET4, "analyse_batch_files", analyse)
    GET4.compare_all_batches(args)
    assert targets == [20.0, 20.0, 20.0]


def test_all_batches_rejects_unknown_physical_scale(tmp_path):
    _image_folders(tmp_path)
    args = _args(tmp_path)
    args.px_size = None
    with pytest.raises(ValueError, match="pixel size"):
        GET4.compare_all_batches(args)


def test_cli_dispatches_grouped_mode_with_all_detectors(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(GET4, "compare_all_batches", lambda args: calls.append(args))
    GET4.main(["--compare-all", str(tmp_path), "--no-plots"])
    assert len(calls) == 1
    assert calls[0].detector is None
    assert calls[0].inputs == [str(tmp_path)]
    assert calls[0].no_plots


@pytest.mark.parametrize("argv", [[], ["--compare-all"], ["--detector", "BSE"]])
def test_cli_uses_project_batches_and_output_by_default(tmp_path, monkeypatch, argv):
    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(GET4, "compare_all_batches", lambda args: calls.append(args))
    GET4.main(argv)
    root = Path(GET4.__file__).resolve().parent.parent
    assert len(calls) == 1
    assert calls[0].inputs == [str(root / "Batches")]
    assert Path(calls[0].out) == root


def test_default_command_generates_json_from_calibrated_images(tmp_path, monkeypatch):
    root = tmp_path / "project"
    batches = root / "Batches"
    batches.mkdir(parents=True)
    _image_folders(batches)
    for path in batches.glob("Batch_*/*.tif"):
        image = tifffile.imread(path)
        rgb = np.repeat(image[..., None], 3, axis=-1)
        tifffile.imwrite(path, rgb, compression="lzw", photometric="rgb",
                         resolution=(1_000_000, 1_000_000), resolutionunit="CENTIMETER")
    se_path = batches / "Batch_2" / "img_se_only_SE.tif"
    se_path.touch()
    monkeypatch.setattr(GET4, "PROJECT_ROOT", root)
    monkeypatch.chdir(tmp_path)
    GET4.main([])
    report_path = root / "all_batch_comparisons.json"
    report = json.loads(report_path.read_text())
    assert set(path.name for path in root.iterdir()) == {"Batches", "all_batch_comparisons.json"}
    assert sum(len(data["comparisons"]) for data in report["detectors"].values()) == 9
    assert report["excluded_images"] == [{
        "batch": "Batch_2", "image": se_path.name, "reason": "SE detector excluded from BSE/ETD/Inlens comparisons",
    }]
    assert all(data["target_pixel_nm"] == 10.0 for data in report["detectors"].values())


def test_se_files_are_excluded_without_opening_them(tmp_path):
    _batch_folders(tmp_path)
    image = tmp_path / "Batch_2" / "img_cell003_SE.tif"
    image.touch()
    excluded = []
    groups = GET4.collect_batch_groups([tmp_path], excluded_images=excluded)
    assert "se" not in groups["Batch_2"]
    assert len(groups["Batch_2"]["bse"]) == 2
    assert excluded[0]["image"] == image.name
    assert excluded[0]["batch"] == "Batch_2"


def test_cli_rejects_conflicting_comparison_modes():
    with pytest.raises(SystemExit) as error:
        GET4.main(["--compare-all", "--compare", "a.json", "b.json"])
    assert error.value.code == 2


@pytest.mark.parametrize("field,value", [("tol_rel", -0.1), ("tol", "pore=nan"), ("tol", "unknown=0.1")])
def test_invalid_tolerances_fail_before_image_analysis(tmp_path, field, value):
    _batch_folders(tmp_path)
    args = _args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="tolerance"):
        GET4.compare_all_batches(args)
    assert not Path(args.out).exists()


def test_legacy_single_batch_cli_still_writes_reports(tmp_path):
    _image_folders(tmp_path)
    out = tmp_path / "single_output"
    GET4.main([str(tmp_path / "Batch_1"), "--out", str(out), "--px-size", "10",
               "--max-lag", "8", "--no-flatten", "--no-destripe", "--no-plots"])
    saved = json.loads((out / "batch_uncertainty.json").read_text())
    assert all(phase["n_images"] == 2 for phase in saved["phases"].values())
    assert all("BSE" in image["image"] for image in saved["images"])
    assert not list(out.glob("*.png"))


def test_plotting_reports_still_work(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    _image_folders(tmp_path)
    path = tmp_path / "Batch_1" / "img_cell0_BSE.tif"
    args = _args(tmp_path)
    args.no_plots = False
    paths = [path]
    GET4.analyse_batch_files(paths, [GET4.read_meta(path)], 10.0, "test scale", args)
    assert (Path(args.out) / "img_cell0_BSE_uncertainty.png").is_file()
    assert (Path(args.out) / "batch_uncertainty.png").is_file()


def test_all_batches_requires_multiple_folders(tmp_path):
    folder = tmp_path / "Batch_1"
    folder.mkdir()
    (folder / "img_cell_BSE.tif").touch()
    with pytest.raises(ValueError, match="at least two"):
        GET4.collect_batch_groups([tmp_path])
