"""
Refresh pubs.bib from Google Scholar.

Weekly / on-demand:
  1. Fetch the Google Scholar profile page.
  2. Diff against the current pubs.bib.
  3. For each new paper, fetch BibTeX from arxiv (search by title), strip to the
     minimal field set (author, title, year, url, note), and append to pubs.bib.
  4. For each existing paper whose Scholar venue is a real conference/journal and
     note={Preprint}, replace the note with the cleaned venue and add journal={venue}.
  5. Write pubs.bib only if it actually changed; let the workflow decide what to do.

The script uses string-level editing so that the user's hand-formatted BibTeX
is preserved; only the affected fields are rewritten.

Failure mode: if Scholar scraping fails after retries, log a warning via
GitHub Actions' ::warning:: annotation and exit 0 so the workflow can still
push a PR on a no-op, or skip cleanly.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import bibtexparser  # type: ignore[import-unresolved]
except ImportError:  # pragma: no cover
    sys.stderr.write("bibtexparser is required. pip install -r .github/scripts/requirements.txt\n")
    raise

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCHOLAR_USER = os.environ.get("SCHOLAR_USER", "rPclf40AAAAJ")
PUBS_BIB_PATH = Path(os.environ.get("PUBS_BIB_PATH", "pubs.bib"))
GITHUB_OUTPUT = os.environ.get("GITHUB_OUTPUT")

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

SCHOLAR_URL = f"https://scholar.google.com/citations?user={SCHOLAR_USER}&hl=en&oi=sra"

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_BIBTEX = "https://arxiv.org/bibtex/{arxiv_id}"

REQ_TIMEOUT = 20  # seconds
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 30  # seconds

NOTE_INDENT = "      "  # 6-space indent for entry fields


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def gh_warning(msg: str) -> None:
    safe = msg.replace("\n", " ").replace("\r", " ")
    print(f"::warning::{safe}", flush=True)


def gh_error(msg: str) -> None:
    safe = msg.replace("\n", " ").replace("\r", " ")
    print(f"::error::{safe}", flush=True)


def set_output(name: str, value: str) -> None:
    if not GITHUB_OUTPUT:
        return
    with open(GITHUB_OUTPUT, "a", encoding="utf-8") as f:
        f.write(f"{name}={value}\n")


# ---------------------------------------------------------------------------
# HTTP helpers with retry
# ---------------------------------------------------------------------------


def http_get(url: str, params: Optional[dict] = None, attempts: int = RETRY_ATTEMPTS) -> Optional[requests.Response]:
    """GET with retries on 429 / 5xx / network errors. Returns None on final failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
                timeout=REQ_TIMEOUT,
            )
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                log(f"[retry] {url} -> {resp.status_code} (attempt {attempt}/{attempts})")
            elif resp.status_code == 200:
                return resp
            else:
                gh_warning(f"Non-retryable HTTP {resp.status_code} fetching {url}")
                return None
        except requests.RequestException as e:
            last_exc = e
            log(f"[retry] {url} -> {type(e).__name__}: {e} (attempt {attempt}/{attempts})")
        if attempt < attempts:
            time.sleep(RETRY_BACKOFF * attempt)  # linear backoff
    gh_warning(f"All {attempts} attempts failed for {url}: {last_exc}")
    return None


# ---------------------------------------------------------------------------
# Parsing — Google Scholar
# ---------------------------------------------------------------------------


ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\d.]+)", re.IGNORECASE)
ARXIV_ID_BRACKET_RE = re.compile(r"\[(?:arXiv:)?(\d{4}\.\d{4,5})\]", re.IGNORECASE)
ARXIV_ID_FROM_ATOM_RE = re.compile(r"<id>\s*http://arxiv\.org/abs/([\d.]+)(v\d+)?\s*</id>")


def parse_scholar(html: str) -> list[dict]:
    """Parse publications from the Scholar profile page.

    Returns a list of dicts: { title, year, venue, arxiv_id, scholar_url }.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []

    for row in soup.select("tr.gsc_a_tr"):
        a = row.select_one("a.gsc_a_at")
        if not a:
            continue
        title = a.get_text(strip=True)
        scholar_url = a.get("href", "")

        year = None
        venue = None
        grays = row.select("div.gs_ibl, div.gs_gray")
        if len(grays) >= 2:
            tail = grays[1].get_text(" ", strip=True)
            m = re.search(r"(.*?)(\d{4})\s*$", tail)
            if m:
                venue = m.group(1).strip().rstrip(",").strip() or None
                year = m.group(2)
            else:
                venue = (tail or None)

        year_span = row.select_one("span.gsc_a_h")
        if year_span and not year:
            year = year_span.get_text(strip=True)

        arxiv_id = None
        m = ARXIV_ID_BRACKET_RE.search(title)
        if m:
            arxiv_id = m.group(1)
        if not arxiv_id:
            m = ARXIV_URL_RE.search(scholar_url)
            if m:
                arxiv_id = m.group(1)

        if not title:
            continue
        out.append({
            "title": title,
            "year": year,
            "venue": venue,
            "arxiv_id": arxiv_id,
            "scholar_url": scholar_url,
        })

    return out


# ---------------------------------------------------------------------------
# Venue cleaning
# ---------------------------------------------------------------------------


def clean_venue(raw: Optional[str]) -> Optional[str]:
    """Convert a raw Scholar venue string into a usable Note value.

    Returns None if the venue is not informative (e.g. arXiv self-reference, blank,
    or just a year).
    """
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None

    # Filter arXiv self-reference (Scholar often says "arXiv preprint arXiv:XXX")
    if re.search(r"arxiv", s, re.IGNORECASE):
        return None

    # Strip "Proceedings of the " (with optional year and "the")
    s = re.sub(r"^Proceedings of (?:the )?(?:\d{4}\s+)?", "", s, flags=re.IGNORECASE)

    # Split on commas and walk through parts, stopping at the first junk token
    parts = [p.strip() for p in s.split(",")]
    name_parts: list[str] = []
    for p in parts:
        is_year = re.fullmatch(r"\d{4}", p) is not None
        is_page = re.fullmatch(r"\d+\s*[-–]\s*\d+", p) is not None
        is_vol_iss = re.fullmatch(r"\d+\s*\(\d+\)", p) is not None
        is_article = re.fullmatch(r"\d{5,}\s*[…]*", p) is not None
        is_just_ellipsis = p == "…"
        if is_year or is_page or is_vol_iss or is_article or is_just_ellipsis:
            break
        name_parts.append(p)

    if not name_parts:
        return None

    # Post-process the last name part: strip trailing volume/issue/article/ellipsis
    last = name_parts[-1]
    last = re.sub(r"\s+\d+\s*\(\d+\)\s*$", "", last)
    last = re.sub(r"\s+\d{1,3}\s*$", "", last)
    last = re.sub(r"\s+\d{5,}\s*$", "", last)
    last = re.sub(r"\s*…\s*$", "", last)
    name_parts[-1] = last

    name = " ".join(name_parts).strip()

    # Strip redundant "-IEEE Conference on Computer Communications" suffix
    name = re.sub(
        r"\s*-\s*IEEE\s+Conference\s+on\s+Computer\s+Communications\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"[\s,…]+$", "", name)
    name = name.strip().rstrip(",").strip()

    if not name or re.fullmatch(r"\d{1,4}", name):
        return None

    return name


# ---------------------------------------------------------------------------
# arxiv helpers
# ---------------------------------------------------------------------------


def arxiv_search_by_title(title: str) -> Optional[str]:
    params = {"search_query": f'ti:"{title}"', "max_results": 1}
    resp = http_get(ARXIV_API, params=params)
    if resp is None:
        return None
    m = ARXIV_ID_FROM_ATOM_RE.search(resp.text)
    return m.group(1) if m else None


def fetch_arxiv_bibtex(arxiv_id: str) -> Optional[str]:
    url = ARXIV_BIBTEX.format(arxiv_id=arxiv_id)
    resp = http_get(url)
    if resp is None:
        return None
    text = resp.text.strip()
    if not text.startswith("@"):
        gh_warning(f"arxiv bibtex endpoint did not return BibTeX for {arxiv_id}")
        return None
    return text


# ---------------------------------------------------------------------------
# Local pubs.bib — parsing & string-level editing
# ---------------------------------------------------------------------------


def normalize_title(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_pubs_bib(text: str) -> list[dict]:
    db = bibtexparser.loads(text)
    return list(db.entries)


def clean_arxiv_bibtex(raw: str, note: str = "Preprint", journal: Optional[str] = None) -> Optional[str]:
    """Strip arxiv bibtex down to the minimal fields the user wants:
    author, title, year, url, note (and optionally journal).

    Returns the cleaned BibTeX entry as a string, or None on parse failure.
    """
    try:
        db = bibtexparser.loads(raw)
    except Exception as e:
        gh_warning(f"arxiv bibtex parse failed: {e}")
        return None
    if not db.entries:
        return None
    entry = db.entries[0]

    entry_type = entry.get("ENTRYTYPE", "misc")
    entry_id = entry.get("ID", "")
    if not entry_id:
        return None

    field_order = ["author", "title", "year", "journal", "url", "note"]
    lines = []
    for f in field_order:
        if f == "note":
            v = note
        elif f == "journal":
            v = journal or ""
        else:
            v = entry.get(f, "")
        if not v:
            continue
        v = v.strip()
        lines.append(f"{NOTE_INDENT}{f}={{{v}}},")

    if not lines:
        return None

    return f"@{entry_type}{{{entry_id},\n" + "\n".join(lines) + "\n}\n"


def _find_entry_span(text: str, entry_id: str) -> Optional[tuple[int, int, str, str, str]]:
    """Locate the entry by citation key. Returns (start, end, head, body, tail) or None."""
    pattern = re.compile(
        r"(@\w+\s*\{\s*" + re.escape(entry_id) + r"\s*,)(.*?)(\n[ \t]*\}\s*(?:\n|$))",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return None
    return m.start(), m.end(), m.group(1), m.group(2), m.group(3)


def _replace_field(body: str, field_name: str, value: str) -> str:
    """Replace or insert a field (note, journal, etc.) in an entry body string."""
    pattern = re.compile(r"(\n[ \t]*" + field_name + r"\s*=\s*\{)([^}]*)(\})", re.IGNORECASE)
    if pattern.search(body):
        return pattern.sub(
            lambda m: m.group(1) + value + m.group(3),
            body,
            count=1,
        )
    stripped = body.rstrip()
    if stripped.endswith(","):
        new_body = stripped + f"\n{NOTE_INDENT}{field_name}={{{value}}},"
    else:
        new_body = stripped + f",\n{NOTE_INDENT}{field_name}={{{value}}},"
    tail_ws = body[len(stripped):]
    return new_body + tail_ws


def update_note_for_entry(text: str, entry_id: str, new_note: str) -> tuple[str, bool]:
    span = _find_entry_span(text, entry_id)
    if not span:
        return text, False
    start, end, head, body, tail = span
    new_body = _replace_field(body, "note", new_note)
    return text[:start] + head + new_body + tail + text[end:], True


def inject_journal_for_entry(text: str, entry_id: str, journal: str) -> tuple[str, bool]:
    span = _find_entry_span(text, entry_id)
    if not span:
        return text, False
    start, end, head, body, tail = span
    new_body = _replace_field(body, "journal", journal)
    return text[:start] + head + new_body + tail + text[end:], True


def append_entry(text: str, raw_new_entry: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text + raw_new_entry.rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    log(f"Refresh pubs.bib from Google Scholar (user={SCHOLAR_USER})")
    log(f"Reading {PUBS_BIB_PATH}")

    if not PUBS_BIB_PATH.exists():
        gh_error(f"{PUBS_BIB_PATH} does not exist")
        return 1

    original_text = PUBS_BIB_PATH.read_text(encoding="utf-8")

    existing_entries = parse_pubs_bib(original_text)
    existing_by_title: dict[str, dict] = {}
    for e in existing_entries:
        et = e.get("title", "")
        if et:
            existing_by_title[normalize_title(et)] = e
    log(f"Found {len(existing_entries)} existing entries")

    log(f"Fetching {SCHOLAR_URL}")
    resp = http_get(SCHOLAR_URL)
    if resp is None:
        gh_warning("Scholar fetch failed; skipping this run.")
        set_output("changed", "false")
        return 0

    scholar_entries = parse_scholar(resp.text)
    log(f"Parsed {len(scholar_entries)} entries from Scholar")
    if not scholar_entries:
        gh_warning("Scholar returned no entries; skipping this run.")
        set_output("changed", "false")
        return 0

    new_entries_raw: list[str] = []
    note_updates: list[tuple[str, str, str]] = []
    skipped_no_venue: list[str] = []
    skipped_no_arxiv: list[str] = []

    for s in scholar_entries:
        key = normalize_title(s["title"])
        if key not in existing_by_title:
            arxiv_id = s.get("arxiv_id") or arxiv_search_by_title(s["title"])
            if not arxiv_id:
                gh_warning(f"Skip (no arxiv): {s['title']}")
                skipped_no_arxiv.append(s["title"])
                continue
            log(f"New: {s['title']} (arxiv {arxiv_id})")
            bib = fetch_arxiv_bibtex(arxiv_id)
            if not bib:
                gh_warning(f"Skip (arxiv bibtex failed): {s['title']} ({arxiv_id})")
                continue
            cleaned = clean_arxiv_bibtex(bib, note="Preprint")
            if cleaned:
                new_entries_raw.append(cleaned)
        else:
            existing = existing_by_title[key]
            entry_id = existing.get("ID", "")
            current_note = (existing.get("note") or "").strip()
            cleaned = clean_venue(s.get("venue"))
            if not cleaned:
                if s.get("venue"):
                    skipped_no_venue.append(f"{s['title']} (Scholar venue: {s['venue']!r})")
                continue
            if current_note.lower() == "preprint":
                log(f"Update note: {s['title']!r} -> {cleaned!r}")
                note_updates.append((entry_id, s["title"], cleaned))
            else:
                log(f"No update (note already {current_note!r}): {s['title']}")

    if skipped_no_venue:
        log(f"Skipped {len(skipped_no_venue)} entries with no usable venue:")
        for x in skipped_no_venue:
            log(f"  - {x}")
    if skipped_no_arxiv:
        log(f"Skipped {len(skipped_no_arxiv)} new entries without arxiv ID")

    if not new_entries_raw and not note_updates:
        log("No changes.")
        set_output("changed", "false")
        return 0

    new_text = original_text
    for entry_id, title, new_note in note_updates:
        if not entry_id:
            continue
        new_text, ok = update_note_for_entry(new_text, entry_id, new_note)
        if not ok:
            gh_warning(f"Could not find entry_id={entry_id!r} in pubs.bib (title: {title})")
        else:
            new_text, ok2 = inject_journal_for_entry(new_text, entry_id, new_note)
            if not ok2:
                gh_warning(f"Could not inject journal for entry_id={entry_id!r}")

    for raw in new_entries_raw:
        new_text = append_entry(new_text, raw)

    if new_text == original_text:
        log("No effective changes (content identical).")
        set_output("changed", "false")
        return 0

    PUBS_BIB_PATH.write_text(new_text, encoding="utf-8")
    log(f"Wrote {PUBS_BIB_PATH} (+{len(new_entries_raw)} new, ~{len(note_updates)} updated)")
    set_output("changed", "true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
