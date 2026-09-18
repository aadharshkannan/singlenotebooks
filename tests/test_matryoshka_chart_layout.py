from xml.etree import ElementTree

from sampling_comparison.matryoshka_report import _render_multi_series_svg


def test_coincident_native_endpoints_use_separate_visible_legend_labels():
    svg = _render_multi_series_svg(
        chart_id="fixture", title="Paired rate deltas", description="Test fixture",
        y_label="Delta (pp)", zero_line=True,
        series=[(f"rate {rate}%", {8: -float(rate), 1536: 0.0}, "#005a9c") for rate in (1, 2, 5, 10, 20)],
    )
    root = ElementTree.fromstring(svg)
    legend = [node for node in root.findall("text") if "legend-label" in node.attrib.get("class", "")]
    assert len(legend) == 5
    assert len({(node.attrib["x"], node.attrib["y"]) for node in legend}) == 5
    assert all(0 < float(node.attrib["x"]) < 500 for node in legend)
    assert 'class="direct-label"' not in svg


def test_chart_renders_every_low_dimension_without_untested_historical_ticks():
    dimensions = [*range(2, 33, 2), 1536]
    svg = _render_multi_series_svg(
        chart_id="low-dimensions", title="Low-dimensional sweep", description="Test fixture",
        y_label="MAE", series=[("MAE", {d: 0.2 + d / 10000 for d in dimensions}, "#005a9c")],
    )
    root = ElementTree.fromstring(svg)
    labels = [node.text for node in root.findall("text")]
    assert all(str(d) in labels for d in dimensions)
    assert all(str(d) not in labels for d in (64, 128, 256, 512))
