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
