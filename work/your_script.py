"""Sucht für Alben aus einer CSV per Gemini + Google-Suche nach Tracklisten
und schreibt die Ergebnisse als JSON-Datei.

Das Skript liegt in <project-root>/work, die CSV in <project-root>/data.

Aufruf (aus dem Projekt-Root):
    python work/your_script.py                     # data/development.csv -> work/results.json
    python work/your_script.py data/andere.csv -o work/ergebnis.json
    python work/your_script.py --limit 3           # nur die ersten 3 Zeilen testen
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv(".env", override=True)

MAX_ATTEMPTS = 3

# Das Skript liegt in <project-root>/work, die CSV in <project-root>/data
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "development.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "results.json"


# --------------------------------------------------------------------------- #
# Datenmodelle
# --------------------------------------------------------------------------- #
class Track(BaseModel):
    disc: int = 1
    number: int
    title: str
    length_sec: Optional[int] = None


class AgentAnswer(BaseModel):
    """Das soll das Modell zurückliefern."""

    tracks: list[Track] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class AlbumResult(BaseModel):
    """Eintrag in der Ausgabedatei (album/artist/barcode kommen unverändert aus der CSV)."""

    album: str
    artist: str
    barcode: str
    tracks: list[Track]
    sources: list[str]


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
PROMPT = """You are a music metadata researcher. Find the track list of one specific release using Google Search.

Release (from our catalogue; it may contain typos or format hints like "(2CD)" or "(LP)"):
- barcode (EAN/UPC): {barcode}
- album title: {album}
- album artist: {artist}

Rules:
1. If a barcode is given, search for it first (MusicBrainz, Discogs, shops, label or artist pages). Then search by artist and title.
2. The barcode identifies the exact edition. Use the edition that matches it (e.g. deluxe vs. standard, 1 vs. 2 discs). Without a barcode, use hints in the title ("(2CD)", "(LP)", ...) to choose the edition.
3. Take tracks ONLY from pages that your search returned. Never write tracks from memory. If you cannot find a track list you trust, return an empty "tracks" list.
4. "Various", "VA" and similar mean a compilation. "Name, Die" means "Die Name".
5. Keep track titles exactly as written on the source and do not translate them. "disc" starts at 1; "number" is the position on that disc (for vinyl, count continuously per record: A1, A2, B1 -> 1, 2, 3). Set "length_sec" to the duration in seconds, or null if the source gives none.
6. "sources" lists the URLs of the pages you took the tracks from.

Answer with a single JSON object and nothing else (no markdown, no comments):
{{"tracks": [{{"disc": 1, "number": 1, "title": "...", "length_sec": 215}}], "sources": ["https://..."]}}
"""


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def parse_answer(text: str) -> AgentAnswer:
    """Holt das JSON-Objekt aus der Modellantwort (auch wenn es in ```-Blöcken steckt)."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("kein JSON-Objekt in der Antwort")
    return AgentAnswer.model_validate_json(cleaned[start : end + 1])


def restructure(client: genai.Client, model: str, text: str) -> AgentAnswer:
    """Fallback: Freitext-Antwort in das Schema bringen (ohne Suchtool, mit erzwungenem JSON)."""
    if not text.strip():
        return AgentAnswer()
    response = client.models.generate_content(
        model=model,
        contents=(
            "Convert the following text into the JSON schema. Use only information "
            "contained in the text. If it contains no track list, return empty lists.\n\n"
            + text
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AgentAnswer,
            temperature=0,
        ),
    )
    return AgentAnswer.model_validate_json(response.text)


def grounding_urls(response) -> list[str]:
    """URLs der Seiten, die Google Search tatsächlich geliefert hat."""
    urls: list[str] = []
    for candidate in response.candidates or []:
        metadata = candidate.grounding_metadata
        for chunk in (metadata.grounding_chunks or []) if metadata else []:
            uri = chunk.web.uri if chunk.web else None
            if uri and uri not in urls:
                urls.append(uri)
    return urls


def research_album(
    client: genai.Client, model: str, barcode: str, album: str, artist: str
) -> AgentAnswer:
    prompt = PROMPT.format(
        barcode=barcode if barcode not in ("", "0") else "unknown",
        album=album,
        artist=artist,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0,
        ),
    )
    text = response.text or ""
    try:
        answer = parse_answer(text)
    except ValueError:
        answer = restructure(client, model, text)

    # Doppelte Tracks entfernen, sortieren
    tracks: dict[tuple[int, int], Track] = {}
    for track in sorted(answer.tracks, key=lambda t: (t.disc, t.number)):
        tracks.setdefault((track.disc, track.number), track)

    if not tracks:
        return AgentAnswer()

    sources = list(dict.fromkeys(answer.sources))
    if not sources:  # Fallback: tatsächlich abgerufene Seiten laut Grounding-Metadaten
        sources = grounding_urls(response)
    return AgentAnswer(tracks=list(tracks.values()), sources=sources)


def lookup_with_retry(
    client: genai.Client, model: str, barcode: str, album: str, artist: str
) -> AgentAnswer:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return research_album(client, model, barcode, album, artist)
        except Exception as exc:  # z. B. Rate Limit oder Netzwerkfehler
            print(f"  Versuch {attempt}/{MAX_ATTEMPTS} fehlgeschlagen: {exc}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt + 1))
    return AgentAnswer()


def read_rows(path: Path) -> list[dict[str, str]]:
    # csv-Modul liest alles als String, dadurch gehen führende Nullen im Barcode nicht verloren
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [
            row
            for row in csv.DictReader(f)
            if any((value or "").strip() for value in row.values())
        ]


def save(results: list[AlbumResult], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV-Datei (Standard: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON-Ausgabedatei (Standard: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--limit", type=int, help="nur die ersten N Zeilen verarbeiten")
    args = parser.parse_args()

    rows = read_rows(args.input)[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[AlbumResult] = []

    with genai.Client(
        vertexai=False,
        api_key=os.environ["GEMINI_API_KEY"],
    ) as client:
        model = os.environ["GOOGLE_GENAI_MODEL"]
        for i, row in enumerate(rows, start=1):
            barcode = (row.get("barcode") or "").strip()
            album = (row.get("album_title") or "").strip()
            artist = (row.get("album_artist") or "").strip()

            answer = lookup_with_retry(client, model, barcode, album, artist)
            results.append(
                AlbumResult(
                    album=album,
                    artist=artist,
                    barcode=barcode,
                    tracks=answer.tracks,
                    sources=answer.sources,
                )
            )
            save(results, args.output)  # nach jedem Album speichern, falls der Lauf abbricht
            print(f"[{i}/{len(rows)}] {artist} - {album}: {len(answer.tracks)} Tracks", flush=True)

    print(f"Fertig: {len(results)} Alben in {args.output}")


if __name__ == "__main__":
    main()
