from app.evaluator import build_evaluation_out, evaluate_candidates
from app.extractor import extract_release
from app.normalizer import normalize_input
from tests.conftest import make_sample_release


def test_evaluator_matched_and_clear_winner_among_duplicate_pressings() -> None:
    norm = normalize_input(
        "4262459550239", "Long Player (RSD Exklusive)", "Fantastischen Vier, Die"
    )
    # Two identical top pressings + one clearly weaker diverging CD edition
    rel_1 = extract_release(make_sample_release(release_id=9900001))
    rel_2 = extract_release(make_sample_release(release_id=9900002, community_have=10))
    rel_weak = extract_release(
        make_sample_release(
            release_id=9900003,
            barcode="0000000000000",
            title="Long Player Deluxe",
            formats=[{"name": "CD", "qty": "1", "descriptions": ["Album"]}],
            notes="",
            extra_track=True,
            community_have=5,
        )
    )

    bundle = evaluate_candidates(
        norm,
        [rel_1, rel_2, rel_weak],
        {
            9900001: "barcode_exact",
            9900002: "barcode_exact",
            9900003: "title_artist_structured",
        },
    )
    assert bundle.status == "matched"
    assert bundle.editions_agree is False
    assert bundle.selected is not None
    assert bundle.selected.discogs_release_id == 9900001
    assert len(bundle.alternatives) == 2

    ev_out = build_evaluation_out(bundle, 3)
    assert ev_out.review_reasons == []
    assert ev_out.duration_coverage == 1.0
    assert ev_out.performer_coverage == 1.0


def test_evaluator_flags_needs_review_when_close_editions_diverge() -> None:
    norm = normalize_input(
        "4262459550239", "Long Player (RSD Exklusive)", "Fantastischen Vier, Die"
    )
    rel_a = extract_release(make_sample_release(release_id=9900001))
    rel_b = extract_release(make_sample_release(release_id=9900002, extra_track=True))

    bundle = evaluate_candidates(
        norm,
        [rel_a, rel_b],
        {9900001: "barcode_exact", 9900002: "barcode_exact"},
    )
    assert bundle.status == "needs_review"
    assert bundle.editions_agree is False
    assert any("distinct tracklist editions" in r for r in bundle.selected_eval.review_reasons)


def test_evaluator_rejects_empty_tracklist_and_barcode_collisions() -> None:
    # Single-act barcode collision
    norm_single = normalize_input("4065629733131", "Aspiral", "EPICA")
    wrong_single = extract_release(
        make_sample_release(
            release_id=9900010,
            title="Completely Unrelated Symphony",
            artist="Some Other Band",
            barcode="4065629733131",
        )
    )
    res_single = evaluate_candidates(norm_single, [wrong_single], {9900010: "barcode_exact"})
    assert res_single.status == "not_found"
    assert res_single.rejection_reasons

    # Compilation barcode collision
    norm_comp = normalize_input("4015698189711", "Krautrock Eruption", "Various")
    wrong_comp = extract_release(
        make_sample_release(
            release_id=9900011,
            title="Tropical Hits 1999",
            artist="Various",
            barcode="4015698189711",
        )
    )
    res_comp = evaluate_candidates(norm_comp, [wrong_comp], {9900011: "barcode_exact"})
    assert res_comp.status == "not_found"

    # Empty tracklist
    empty_rel = extract_release(make_sample_release(release_id=9900012, tracklist=[]))
    res_empty = evaluate_candidates(norm_single, [empty_rel], {9900012: "barcode_exact"})
    assert res_empty.status == "not_found"


def test_evaluator_flags_quality_disc_mismatch_and_unconfirmed_barcode() -> None:
    norm = normalize_input("4029759202219", "JETHRO TULL;BALOISE SESSION(2CD)", "JETHRO TULL")
    rel = extract_release(
        make_sample_release(
            release_id=9900020,
            title="Baloise Session",
            artist="Jethro Tull",
            barcode="",
            formats=[{"name": "CD", "qty": "2", "descriptions": ["Album"]}],
            data_quality="Needs Major Changes",
            notes="",
            tracklist=[
                {"type_": "track", "position": "1", "title": "Song 1", "duration": "3:00"},
                {"type_": "track", "position": "2", "title": "Song 2", "duration": "3:10"},
            ],
        )
    )
    bundle = evaluate_candidates(norm, [rel], {9900020: "title_artist_structured"})
    assert bundle.status == "needs_review"
    assert bundle.selected_eval is not None
    assert bundle.selected_eval.disc_format_consistent is False
    reasons = " | ".join(bundle.selected_eval.review_reasons)
    assert "Supplied barcode was not confirmed" in reasons
    assert "Needs Major Changes" in reasons
    assert "disc count" in reasons


def test_evaluator_matches_label_catno_titles_and_rejects_wrong_catno_number() -> None:
    comp_tracks = [
        {
            "type_": "track",
            "position": "A1",
            "title": "Track One",
            "duration": "6:03",
            "artists": [{"id": 701, "name": "Dan Piu"}],
        },
        {
            "type_": "track",
            "position": "B1",
            "title": "Track Two",
            "duration": "4:32",
            "artists": [{"id": 702, "name": "Rinaldo Makaj"}],
        },
    ]
    norm_432 = normalize_input("", "432HERTZ 002", "VA")
    rel_432 = make_sample_release(
        release_id=35007128,
        title="303rd Dimensions",
        artist="Various",
        barcode="",
        formats=[{"name": "Vinyl", "qty": "1", "descriptions": ['12"']}],
        notes="",
        tracklist=comp_tracks,
    )
    rel_432["labels"] = [{"name": "432HERTZ BERLIN", "catno": "432-002"}]
    bundle_432 = evaluate_candidates(
        norm_432, [extract_release(rel_432)], {35007128: "label_catno_structured"}
    )
    assert bundle_432.status == "matched"
    assert bundle_432.selected is not None
    assert bundle_432.selected.discogs_release_id == 35007128

    norm_jf = normalize_input("0", "Jackfruit Traxx 1", "Various Artists")
    rel_jf1 = make_sample_release(
        release_id=36475918,
        title="Traxx Salad Vol. 1",
        artist="Various",
        barcode="",
        formats=[{"name": "Vinyl", "qty": "1", "descriptions": ['12"']}],
        notes="",
        tracklist=comp_tracks,
    )
    rel_jf1["labels"] = [{"name": "JackFruit Traxx", "catno": "JFXX1"}]
    rel_jf2 = make_sample_release(
        release_id=37837974,
        title="Traxx Salad Vol. 2",
        artist="Various",
        barcode="",
        formats=[{"name": "Vinyl", "qty": "1", "descriptions": ['12"']}],
        notes="",
        tracklist=comp_tracks,
    )
    rel_jf2["labels"] = [{"name": "JackFruit Traxx", "catno": "JFXX2"}]
    bundle_jf = evaluate_candidates(
        norm_jf,
        [extract_release(rel_jf2), extract_release(rel_jf1)],
        {37837974: "freetext_q", 36475918: "label_catno_structured"},
    )
    assert bundle_jf.status == "matched"
    assert bundle_jf.selected is not None
    assert bundle_jf.selected.discogs_release_id == 36475918
    assert bundle_jf.alternatives == []
