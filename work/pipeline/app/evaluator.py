import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

from app.extractor import ExtractedRelease
from app.models import AlternativeEditionOut, EvaluationOut, NormalizedInput
from app.normalizer import extract_label_catno_hint, fold_accents, normalize_text

NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
EP_WORD_RE = re.compile(r"\bep\b")
CATNO_TRAILING_NUM_RE = re.compile(r"(\d+)\s*$")
LOW_QUALITY_GRADES = {"Needs Major Changes", "Entirely Incorrect"}


@dataclass
class CandidateEvaluation:
    discogs_release_id: int
    score: float
    match_method: str
    barcode_matched: bool
    title_sim: float
    artist_sim: float
    format_matched: bool
    disc_format_consistent: bool
    track_count: int
    duration_coverage: float
    performer_coverage: float
    writer_credit_coverage: float
    producer_credit_coverage: float
    review_reasons: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    passed: bool = False


@dataclass
class EvaluationBundle:
    status: Literal["matched", "needs_review", "not_found"]
    selected: ExtractedRelease | None
    selected_eval: CandidateEvaluation | None
    ranked: list[tuple[ExtractedRelease, CandidateEvaluation]]
    alternatives: list[AlternativeEditionOut]
    editions_agree: bool
    rejection_reasons: list[str]


def _clean_for_similarity(text: str) -> str:
    folded = fold_accents(normalize_text(text))
    cleaned = NON_ALNUM_RE.sub(" ", folded)
    return re.sub(r"\s+", " ", cleaned).strip()


def string_similarity(a: str, b: str) -> float:
    ca = _clean_for_similarity(a)
    cb = _clean_for_similarity(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    seq = SequenceMatcher(None, ca, cb).ratio()
    tok_a = set(ca.split())
    tok_b = set(cb.split())
    jaccard = len(tok_a & tok_b) / len(tok_a | tok_b) if (tok_a and tok_b) else 0.0
    # Containment bonus when one title contains a clean subtitle of the other
    contain = 0.0
    if len(ca) >= 4 and len(cb) >= 4 and (ca in cb or cb in ca):
        contain = 0.85
    return max(seq, jaccard, contain)


def _title_similarity(input_title: str, candidate_title: str) -> float:
    base = string_similarity(input_title, candidate_title)
    no_ep_in = EP_WORD_RE.sub("", _clean_for_similarity(input_title)).strip()
    no_ep_cand = EP_WORD_RE.sub("", _clean_for_similarity(candidate_title)).strip()
    if no_ep_in and no_ep_cand:
        base = max(base, string_similarity(no_ep_in, no_ep_cand))
    return base


def _format_alignment(
    norm: NormalizedInput, release: ExtractedRelease
) -> tuple[float, bool, bool]:
    """Compare requested format_hint & release formats against inferred track discs."""
    hint = norm.format_hint
    rel_formats = release.formats or []
    rel_names = {str(f.get("name") or "").strip().lower() for f in rel_formats}
    rel_descs: set[str] = set()
    rel_text_blobs: list[str] = [release.title.lower(), release.notes.lower()]
    total_qty = 0

    for f in rel_formats:
        try:
            total_qty += int(f.get("qty") or 1)
        except (TypeError, ValueError):
            total_qty += 1
        for desc in f.get("descriptions") or []:
            rel_descs.add(str(desc).strip().lower())
        if f.get("text"):
            rel_text_blobs.append(str(f["text"]).lower())

    inferred_discs = max((t.disc_number for t in release.tracks), default=1)
    disc_consistent = True
    if total_qty > 1 and inferred_discs < total_qty and any(
        n in {"cd", "vinyl"} for n in rel_names
    ):
        disc_consistent = False
    if hint.qty and hint.qty > 1 and inferred_discs < hint.qty:
        disc_consistent = False

    if not hint.medium and not hint.qty and not hint.tags:
        return 0.5, False, disc_consistent

    checks = 0
    hits = 0
    if hint.medium:
        checks += 1
        if hint.medium.lower() in rel_names:
            hits += 1
    if hint.qty:
        checks += 1
        if total_qty == hint.qty:
            hits += 1
    for tag in hint.tags:
        tag_low = tag.lower()
        if tag_low in {"2cd", "3cd", "cd", "2lp", "3lp", "lp", "vinyl"}:
            continue
        checks += 1
        if "rsd" in tag_low:
            if any("rsd" in blob or "record store day" in blob for blob in rel_text_blobs) or any(
                "record store day" in d or "rsd" in d for d in rel_descs
            ):
                hits += 1
        elif tag_low == "ep":
            if "ep" in rel_descs or " ep" in release.title.lower():
                hits += 1
        elif any(tag_low in blob for blob in rel_text_blobs) or tag_low in rel_descs:
            hits += 1

    if checks == 0:
        return 0.5, False, disc_consistent
    score = hits / checks
    return score, hits == checks, disc_consistent


def _matches_label_catno(norm: NormalizedInput, release: ExtractedRelease) -> bool:
    """Match inputs like '432HERTZ 002' or 'Jackfruit Traxx 1' against release label + catno."""
    hint = extract_label_catno_hint(norm.title_clean)
    if hint is None:
        return False
    prefix, _raw_num, target_int = hint
    prefix_clean = _clean_for_similarity(prefix)
    if not prefix_clean:
        return False

    for lbl in release.labels or []:
        lbl_name = _clean_for_similarity(str(lbl.get("name") or ""))
        catno = str(lbl.get("catno") or "").strip()
        if not lbl_name or not catno:
            continue
        label_match = string_similarity(prefix_clean, lbl_name) >= 0.80 or (
            len(prefix_clean) >= 4
            and (prefix_clean in lbl_name or lbl_name in prefix_clean)
        )
        if not label_match:
            continue
        num_match = CATNO_TRAILING_NUM_RE.search(catno)
        if num_match is not None and int(num_match.group(1)) == target_int:
            return True
    return False


def evaluate_candidate(
    norm: NormalizedInput,
    release: ExtractedRelease,
    match_method: str,
) -> CandidateEvaluation:
    rejection_reasons: list[str] = []
    review_reasons: list[str] = []

    if release.track_count == 0:
        rejection_reasons.append(
            f"Release {release.discogs_release_id} has an empty tracklist."
        )
        return CandidateEvaluation(
            discogs_release_id=release.discogs_release_id,
            score=0.0,
            match_method=match_method,
            barcode_matched=False,
            title_sim=0.0,
            artist_sim=0.0,
            format_matched=False,
            disc_format_consistent=True,
            track_count=0,
            duration_coverage=0.0,
            performer_coverage=0.0,
            writer_credit_coverage=0.0,
            producer_credit_coverage=0.0,
            rejection_reasons=rejection_reasons,
            passed=False,
        )

    cand_barcodes = {b for b, _ in release.barcode_entries}
    input_barcodes = set(norm.barcode_variants)
    barcode_matched = bool(input_barcodes and (cand_barcodes & input_barcodes))

    title_sim = _title_similarity(norm.title_clean, release.title) if norm.title_clean else 0.5

    if norm.is_compilation:
        artist_sim = 1.0 if release.artist_norm == "various" else 0.25
    elif not norm.artist_norm:
        artist_sim = 0.5
    else:
        artist_sim = string_similarity(norm.artist_clean, release.artist_display)

    format_score, format_matched, disc_consistent = _format_alignment(norm, release)

    # Track quality metrics
    total_tracks = release.track_count
    duration_cov = sum(1 for t in release.tracks if t.has_valid_duration) / total_tracks
    performer_cov = sum(1 for t in release.tracks if t.has_resolved_performer) / total_tracks
    writer_cov = (
        sum(
            1
            for t in release.tracks
            if any(c.role_category in {"composer", "lyricist"} for c in t.credits)
        )
        / total_tracks
    )
    producer_cov = (
        sum(1 for t in release.tracks if any(c.role_category == "producer" for c in t.credits))
        / total_tracks
    )

    tie_breaker = min(0.05, math.log10(max(1, release.community_have + 1)) * 0.015)
    if release.data_quality in {"Complete and Correct", "Correct"}:
        tie_breaker += 0.02

    if barcode_matched:
        if norm.title_clean and (
            (norm.is_compilation and title_sim < 0.45)
            or (not norm.is_compilation and title_sim < 0.25 and artist_sim < 0.25)
        ):
            rejection_reasons.append(
                f"Release {release.discogs_release_id} matched barcode but contradicted title/artist "
                f"(title_sim={title_sim:.2f}, artist_sim={artist_sim:.2f})."
            )
            score = 0.2
        else:
            score = min(
                1.0,
                0.55 + 0.20 * title_sim + 0.15 * artist_sim + 0.10 * format_score + tie_breaker,
            )
    else:
        label_catno_matched = _matches_label_catno(norm, release)
        artist_ok_for_catno = (
            (norm.is_compilation and artist_sim >= 0.55)
            or (not norm.is_compilation and (not norm.artist_norm or artist_sim >= 0.55))
        )
        if label_catno_matched and artist_ok_for_catno:
            score = min(
                1.0,
                0.72 + 0.15 * artist_sim + 0.08 * format_score + tie_breaker,
            )
        else:
            if norm.is_compilation and title_sim < 0.65:
                rejection_reasons.append(
                    f"Release {release.discogs_release_id} compilation title similarity too low ({title_sim:.2f})."
                )
            elif not norm.is_compilation and (
                title_sim < 0.55 or (norm.artist_norm and artist_sim < 0.55)
            ):
                rejection_reasons.append(
                    f"Release {release.discogs_release_id} rejected (title_sim={title_sim:.2f}, artist_sim={artist_sim:.2f})."
                )
            score = min(
                1.0,
                0.52 * title_sim + 0.36 * artist_sim + 0.12 * format_score + tie_breaker,
            )

    passed = not rejection_reasons and score >= 0.65
    if not passed and not rejection_reasons:
        rejection_reasons.append(
            f"Release {release.discogs_release_id} overall confidence {score:.2f} below threshold 0.65."
        )

    if norm.barcode_norm and not barcode_matched:
        review_reasons.append("Supplied barcode was not confirmed in release identifiers.")
    if release.data_quality in LOW_QUALITY_GRADES:
        review_reasons.append(f"Discogs data_quality is '{release.data_quality}'.")
    if performer_cov < 1.0:
        review_reasons.append(
            f"Performing artists missing/unresolved on {int(round((1.0 - performer_cov) * total_tracks))} track(s)."
        )
    if not disc_consistent:
        review_reasons.append(
            "Inferred disc count in tracklist does not match release format quantity."
        )
    if norm.format_hint.tags and not format_matched:
        review_reasons.append(
            f"Requested format hints ({', '.join(norm.format_hint.tags)}) not fully confirmed on matched edition."
        )

    return CandidateEvaluation(
        discogs_release_id=release.discogs_release_id,
        score=round(score, 4),
        match_method=match_method,
        barcode_matched=barcode_matched,
        title_sim=round(title_sim, 4),
        artist_sim=round(artist_sim, 4),
        format_matched=format_matched,
        disc_format_consistent=disc_consistent,
        track_count=total_tracks,
        duration_coverage=round(duration_cov, 4),
        performer_coverage=round(performer_cov, 4),
        writer_credit_coverage=round(writer_cov, 4),
        producer_credit_coverage=round(producer_cov, 4),
        review_reasons=review_reasons,
        rejection_reasons=rejection_reasons,
        passed=passed,
    )


def evaluate_candidates(
    norm: NormalizedInput,
    candidates: list[ExtractedRelease],
    match_methods: dict[int, str],
) -> EvaluationBundle:
    evaluated: list[tuple[ExtractedRelease, CandidateEvaluation]] = []
    all_rejections: list[str] = []

    for rel in candidates:
        method = match_methods.get(rel.discogs_release_id, "search")
        ev = evaluate_candidate(norm, rel, method)
        evaluated.append((rel, ev))
        all_rejections.extend(ev.rejection_reasons)

    passing = [(rel, ev) for rel, ev in evaluated if ev.passed]
    passing.sort(key=lambda pair: (pair[1].score, pair[0].community_have), reverse=True)

    if not passing:
        evaluated.sort(key=lambda pair: pair[1].score, reverse=True)
        return EvaluationBundle(
            status="not_found",
            selected=None,
            selected_eval=None,
            ranked=evaluated,
            alternatives=[],
            editions_agree=True,
            rejection_reasons=all_rejections or ["No candidate releases returned by catalog."],
        )

    selected_rel, selected_ev = passing[0]
    unique_hashes = {rel.tracklist_hash for rel, _ in passing}
    editions_agree = len(unique_hashes) == 1

    review_reasons = list(selected_ev.review_reasons)
    alternatives: list[AlternativeEditionOut] = []

    for alt_rel, alt_ev in passing[1:]:
        diff_msg = (
            "Identical tracklist"
            if alt_rel.tracklist_hash == selected_rel.tracklist_hash
            else f"Different tracklist ({alt_rel.track_count} tracks vs {selected_rel.track_count})"
        )
        alternatives.append(
            AlternativeEditionOut(
                discogs_release_id=alt_rel.discogs_release_id,
                url=alt_rel.url,
                title=alt_rel.title,
                artist_display=alt_rel.artist_display,
                year=alt_rel.year,
                country=alt_rel.country,
                formats=alt_rel.formats,
                track_count=alt_rel.track_count,
                tracklist_hash=alt_rel.tracklist_hash,
                score=alt_ev.score,
                match_method=alt_ev.match_method,
                diff_summary=diff_msg,
            )
        )

    if not editions_agree:
        diverging_scores = [
            ev.score
            for rel, ev in passing[1:]
            if rel.tracklist_hash != selected_rel.tracklist_hash
        ]
        top_diverging_score = max(diverging_scores)
        clear_winner = (
            selected_ev.barcode_matched
            and selected_ev.format_matched
            and (selected_ev.score - top_diverging_score >= 0.15)
        )
        if not clear_winner:
            review_reasons.append(
                f"{len(unique_hashes)} distinct tracklist editions passed evaluation; auto-selected highest-scoring edition."
            )

    selected_ev.review_reasons = list(dict.fromkeys(review_reasons))
    status: Literal["matched", "needs_review"] = (
        "needs_review" if selected_ev.review_reasons else "matched"
    )

    return EvaluationBundle(
        status=status,
        selected=selected_rel,
        selected_eval=selected_ev,
        ranked=passing,
        alternatives=alternatives,
        editions_agree=editions_agree,
        rejection_reasons=all_rejections,
    )


def build_evaluation_out(
    bundle: EvaluationBundle, total_checked: int
) -> EvaluationOut:
    assert bundle.selected_eval is not None
    assert bundle.selected is not None
    ev = bundle.selected_eval
    return EvaluationOut(
        match_confidence=ev.score,
        match_method=ev.match_method,
        candidates_checked=total_checked,
        candidates_passing=len(bundle.ranked),
        editions_agree=bundle.editions_agree,
        track_count=ev.track_count,
        duration_coverage=ev.duration_coverage,
        performer_coverage=ev.performer_coverage,
        writer_credit_coverage=ev.writer_credit_coverage,
        producer_credit_coverage=ev.producer_credit_coverage,
        disc_format_consistent=ev.disc_format_consistent,
        discogs_data_quality=bundle.selected.data_quality,
        review_reasons=ev.review_reasons,
        rejection_reasons=bundle.rejection_reasons,
    )
