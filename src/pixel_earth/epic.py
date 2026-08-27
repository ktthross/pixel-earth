"""Fetch DSCOVR/EPIC full-disc imagery into a local mirror of the NASA archive.

``epic.gsfc.nasa.gov`` publishes no rate limit, no API key, and a ``robots.txt``
of bare ``Disallow:`` with no ``Crawl-delay``. Nothing upstream will throttle us,
so every restraint here is self-imposed:

* requests are serialised through a :class:`RateLimiter` with a minimum interval;
* a complete local file costs **zero** requests -- not even a ``HEAD``;
* revalidation is opt-in and conditional (``If-None-Match`` -> 304, no body);
* a byte ceiling is checked before the first body is requested, and again as
  bytes land;
* retries back off exponentially and honour ``Retry-After``.

Downloads mirror the archive path verbatim, so the tree this writes is directly
consumable by :mod:`pixel_earth.batch`::

    data/epic/natural/2024/06/01/png/epic_1b_20240601004554.png
    data/epic/natural/2024/06/01/metadata.json

The mirror is the expensive, polite-to-acquire resource; ``outputs/`` holds the
disposable derived artifacts. Re-segmenting with new settings never touches the
network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

BASE_URL = "https://epic.gsfc.nasa.gov"

# Deliberately identifiable, so an operator can complain instead of blocking.
# No contact address is baked in -- pass --user-agent to add one.
USER_AGENT = "pixel-earth/0.1 (batch archival; python-httpx)"

# Filename prefix per collection, from the documented archive schema.
COLLECTION_PREFIXES = {
    "natural": "epic_1b_",
    "enhanced": "epic_RGB_",
    "aerosol": "epic_uvai_",
    "cloud": "epic_cloudfraction_",
}

# Archive subdirectory -> file extension. "thumbs" holds jpgs, not "thumbs" files.
FORMAT_EXTENSIONS = {"png": "png", "jpg": "jpg", "thumbs": "jpg"}

# Measured medians, used only to project a total when HEAD probing is skipped.
NOMINAL_BYTES = {"png": 2_650_000, "jpg": 213_000, "thumbs": 6_000}

RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

# The 14 digits at the end of every image name: YYYYMMDDHHMMSS.
_IMAGE_STAMP = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$")

_CACHE_DIR = ".cache"
_AVAILABLE_TTL = 6 * 3600  # the list grows daily
_TODAY_METADATA_TTL = 3600  # today's frames are still being added


class EpicError(RuntimeError):
    """Any unrecoverable problem talking to the EPIC archive."""


class BudgetExceeded(EpicError):
    """The projected download would breach the configured byte ceiling."""


# ---------------------------------------------------------------- rate limiting


class RateLimiter:
    """Enforce a minimum interval between requests.

    ``clock`` and ``sleep`` are injected so tests can drive this with a fake
    clock instead of actually waiting.
    """

    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = max(0.0, min_interval)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None and self.min_interval > 0:
            remaining = self._last + self.min_interval - now
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last = now


# ------------------------------------------------------------------ data model


@dataclass(frozen=True)
class Frame:
    """One EPIC image, as described by the metadata API."""

    collection: str
    image: str  # e.g. "epic_1b_20240601004554", no extension
    archive_day: date  # from the image name -- this is what the URL path uses
    captured: datetime | None  # the metadata "date" field, a different instant
    lat: float | None
    lon: float | None

    def url(self, fmt: str) -> str:
        day = self.archive_day
        return (
            f"{BASE_URL}/archive/{self.collection}"
            f"/{day.year:04d}/{day.month:02d}/{day.day:02d}"
            f"/{fmt}/{self.image}.{FORMAT_EXTENSIONS[fmt]}"
        )

    def local_path(self, root: Path, fmt: str) -> Path:
        day = self.archive_day
        return (
            root
            / self.collection
            / f"{day.year:04d}"
            / f"{day.month:02d}"
            / f"{day.day:02d}"
            / fmt
            / f"{self.image}.{FORMAT_EXTENSIONS[fmt]}"
        )


@dataclass(frozen=True)
class Selection:
    """A requested date and the available date it snapped to."""

    requested: date
    resolved: date

    @property
    def snap_days(self) -> int:
        return (self.resolved - self.requested).days

    @property
    def snapped(self) -> bool:
        return self.resolved != self.requested


@dataclass
class Plan:
    """What a fetch would do, computed before any body is downloaded."""

    collection: str
    fmt: str
    selections: list[Selection]
    frames_by_day: dict[date, list[Frame]] = field(default_factory=dict)
    present: list[Frame] = field(default_factory=list)
    missing: list[Frame] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)  # image name -> bytes
    mirror_bytes: int = 0
    probed: bool = False

    @property
    def total_frames(self) -> int:
        return len(self.present) + len(self.missing)

    @property
    def download_bytes(self) -> int:
        """Bytes still to fetch. Exact when :attr:`probed`, else projected."""
        nominal = NOMINAL_BYTES.get(self.fmt, 0)
        return sum(self.sizes.get(f.image, nominal) for f in self.missing)


@dataclass
class FetchStats:
    downloaded: int = 0
    skipped: int = 0  # already complete locally
    current: int = 0  # revalidated, unchanged (304)
    failed: int = 0
    bytes_downloaded: int = 0
    stopped_early: bool = False
    errors: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ HTTP client


class EpicClient:
    """Rate-limited, retrying HTTP access to the EPIC archive."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        delay: float = 0.5,
        retries: int = 5,
        backoff: float = 1.0,
        max_backoff: float = 60.0,
        user_agent: str = USER_AGENT,
        timeout: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": user_agent}
        )
        self._owns_client = client is None
        self.limiter = RateLimiter(delay, clock=clock, sleep=sleep)
        self.retries = max(1, retries)
        self.backoff = backoff
        self.max_backoff = max_backoff
        self._sleep = sleep
        self._jitter = jitter
        self.request_count = 0

    # -- plumbing ----------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> EpicClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _backoff_delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Honour ``Retry-After`` when the server sends one, else back off."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.strip().isdigit():
                return min(float(retry_after), self.max_backoff)
        return min(self.backoff * (2**attempt) * (1.0 + self._jitter()), self.max_backoff)

    def attempt(self, operation: Callable[[], object], *, what: str) -> object:
        """Run ``operation`` with rate limiting and retries.

        ``operation`` must perform one complete attempt and raise
        :class:`_Retry` to request another. Wrapping the *whole* attempt (rather
        than just the request) is what lets a streamed download retry cleanly.
        """
        last_error = ""
        for attempt_index in range(self.retries):
            self.limiter.wait()
            self.request_count += 1
            try:
                return operation()
            except _Retry as retry:
                last_error = retry.reason
                if attempt_index == self.retries - 1:
                    break
                self._sleep(self._backoff_delay(attempt_index, retry.response))
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt_index == self.retries - 1:
                    break
                self._sleep(self._backoff_delay(attempt_index, None))
        raise EpicError(f"{what} failed after {self.retries} attempts: {last_error}")

    # -- requests ----------------------------------------------------------

    def get_json(self, path: str) -> object:
        url = f"{BASE_URL}{path}"

        def once() -> object:
            response = self._client.get(url)
            if response.status_code in RETRY_STATUSES:
                raise _Retry(f"HTTP {response.status_code}", response)
            if response.status_code != 200:
                raise EpicError(f"GET {url} returned HTTP {response.status_code}")
            try:
                return response.json()
            except ValueError as exc:
                raise _Retry(f"invalid JSON: {exc}", response) from exc

        return self.attempt(once, what=f"GET {path}")

    def content_length(self, url: str) -> int | None:
        """HEAD a URL for its exact size. No body is transferred."""

        def once() -> int | None:
            response = self._client.head(url)
            if response.status_code in RETRY_STATUSES:
                raise _Retry(f"HTTP {response.status_code}", response)
            if response.status_code != 200:
                return None
            raw = response.headers.get("Content-Length")
            return int(raw) if raw and raw.isdigit() else None

        return self.attempt(once, what=f"HEAD {url}")  # type: ignore[return-value]

    def download(
        self,
        url: str,
        target: Path,
        *,
        etag: str | None = None,
        resume: bool = True,
    ) -> tuple[str, int, str | None]:
        """Download ``url`` to ``target``, atomically.

        Returns ``(status, bytes_written, etag)`` where status is ``downloaded``
        or ``current`` (a 304 from conditional revalidation).

        The body streams into ``<target>.part``; only once its length matches
        the advertised size is it renamed. A file at the final path is therefore
        complete by construction, and an interrupted transfer resumes from the
        ``.part`` via a Range request.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")

        def once() -> tuple[str, int, str | None]:
            headers: dict[str, str] = {}
            offset = part.stat().st_size if (resume and part.exists()) else 0
            if offset:
                headers["Range"] = f"bytes={offset}-"
            if etag:
                headers["If-None-Match"] = etag

            with self._client.stream("GET", url, headers=headers) as response:
                if response.status_code == 304:
                    return "current", 0, etag
                if response.status_code in RETRY_STATUSES:
                    response.read()
                    raise _Retry(f"HTTP {response.status_code}", response)
                if response.status_code not in (200, 206):
                    response.read()
                    raise EpicError(f"GET {url} returned HTTP {response.status_code}")

                appending = response.status_code == 206 and offset > 0
                declared = response.headers.get("Content-Length")
                body_bytes = int(declared) if declared and declared.isdigit() else None
                expected = (offset + body_bytes) if (appending and body_bytes) else body_bytes

                written = offset if appending else 0
                with open(part, "ab" if appending else "wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
                        written += len(chunk)

                new_etag = response.headers.get("ETag") or etag

            if expected is not None and written != expected:
                # Keep the .part so the next run can resume; never rename a
                # short file into place.
                raise _Retry(f"truncated: got {written} of {expected} bytes", None)

            os.replace(part, target)
            return "downloaded", written - offset, new_etag

        return self.attempt(once, what=f"GET {url}")  # type: ignore[return-value]


class _Retry(Exception):
    """Internal signal that an attempt should be retried."""

    def __init__(self, reason: str, response: httpx.Response | None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.response = response


# ------------------------------------------------------------------- metadata


def _cache_path(root: Path, name: str) -> Path:
    return root / _CACHE_DIR / name


def _is_fresh(path: Path, ttl: float) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) < ttl
    except OSError:
        return False


def available_dates(
    client: EpicClient,
    collection: str,
    *,
    root: Path,
    write: bool = True,
    ttl: float = _AVAILABLE_TTL,
) -> list[date]:
    """Every date with imagery, newest last. Cached locally with a TTL."""
    cache = _cache_path(root, f"available-{collection}.json")
    if _is_fresh(cache, ttl):
        raw = json.loads(cache.read_text())
    else:
        raw = client.get_json(f"/api/{collection}/available")
        if not isinstance(raw, list):
            raise EpicError(f"/api/{collection}/available did not return a list")
        if write:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw))

    days = []
    for entry in raw:
        try:
            days.append(date.fromisoformat(str(entry)[:10]))
        except ValueError:
            continue
    return sorted(set(days))


def _day_dir(root: Path, collection: str, day: date) -> Path:
    return root / collection / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


def parse_frames(raw: Iterable[dict], collection: str) -> list[Frame]:
    """Turn an API response into frames, skipping entries we cannot place.

    The archive directory comes from the timestamp in the image *name*, not the
    ``date`` field -- for 2024-06-01 those differ (00:45:54 vs 00:41:06), and
    only the name matches the URL path.
    """
    frames = []
    for entry in raw:
        name = str(entry.get("image", "")).strip()
        stamp = _IMAGE_STAMP.search(name)
        if not stamp:
            continue
        year, month, day, hour, minute, second = (int(g) for g in stamp.groups())
        try:
            archive_day = date(year, month, day)
        except ValueError:
            continue

        captured = None
        if isinstance(entry.get("date"), str):
            try:
                captured = datetime.fromisoformat(entry["date"])
            except ValueError:
                captured = datetime(year, month, day, hour, minute, second)

        centroid = entry.get("centroid_coordinates") or {}
        frames.append(
            Frame(
                collection=collection,
                image=name,
                archive_day=archive_day,
                captured=captured,
                lat=centroid.get("lat"),
                lon=centroid.get("lon"),
            )
        )
    return frames


def frames_for_date(
    client: EpicClient,
    collection: str,
    day: date,
    *,
    root: Path,
    write: bool = True,
) -> list[Frame]:
    """Frames for one day, cached beside the images as ``metadata.json``.

    Past days are immutable, so their cache never expires; today's gets a short
    TTL because frames are still arriving. An empty response means "no imagery
    that day" -- a normal outcome, not an error: the API returns ``[]`` rather
    than 404 for the archive's real gaps.
    """
    cache = _day_dir(root, collection, day) / "metadata.json"
    is_past = day < datetime.now(timezone.utc).date()
    if cache.exists() and (is_past or _is_fresh(cache, _TODAY_METADATA_TTL)):
        raw = json.loads(cache.read_text())
    else:
        raw = client.get_json(f"/api/{collection}/date/{day.isoformat()}")
        if not isinstance(raw, list):
            raise EpicError(f"/api/{collection}/date/{day} did not return a list")
        if write:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw, indent=1))

    return parse_frames(raw, collection)


# ---------------------------------------------------------------- date picking


def pick_spread(available: Iterable[date], start: date, end: date, count: int) -> list[Selection]:
    """``count`` dates evenly spaced over ``[start, end]``, snapped to available.

    Snapping matters: the archive has a 230-day hole (2019-06-27 -> 2020-02-12,
    DSCOVR safe hold) and an 86-day one in 2025. A naive even split lands inside
    those and, because the API answers ``[]`` rather than 404, would silently
    yield nothing. Each :class:`Selection` reports how far it moved.
    """
    pool = sorted(d for d in available if start <= d <= end)
    if not pool or count <= 0:
        return []
    if count >= len(pool):
        return [Selection(d, d) for d in pool]

    span = (end - start).days
    if count == 1:
        targets = [start + timedelta(days=span // 2)]
    else:
        targets = [start + timedelta(days=round(span * i / (count - 1))) for i in range(count)]

    used: set[date] = set()
    picks = []
    for target in targets:
        nearest = min(
            (d for d in pool if d not in used),
            key=lambda d: (abs((d - target).days), d),
        )
        used.add(nearest)
        picks.append(Selection(requested=target, resolved=nearest))
    return sorted(picks, key=lambda s: s.resolved)


def pick_last(available: Iterable[date], count: int) -> list[Selection]:
    """The ``count`` most recent available dates."""
    pool = sorted(available)
    chosen = pool[-count:] if count > 0 else []
    return [Selection(d, d) for d in chosen]


def pick_explicit(available: Iterable[date], wanted: Iterable[date]) -> list[Selection]:
    """Requested dates, each snapped to the nearest available one."""
    pool = sorted(available)
    if not pool:
        return []
    picks = [
        Selection(requested=day, resolved=min(pool, key=lambda d: (abs((d - day).days), d)))
        for day in wanted
    ]
    return sorted(picks, key=lambda s: s.resolved)


# -------------------------------------------------------------------- planning


def mirror_bytes(root: Path) -> int:
    """Total size of the local mirror, ignoring partial downloads."""
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p.suffix != ".part")


def build_plan(
    client: EpicClient,
    *,
    root: Path,
    collection: str,
    fmt: str,
    selections: list[Selection],
    probe: bool = True,
    write_metadata: bool = True,
    force: bool = False,
) -> Plan:
    """Resolve selections to frames and work out exactly what is missing."""
    plan = Plan(collection=collection, fmt=fmt, selections=selections)
    plan.mirror_bytes = mirror_bytes(root)

    for selection in selections:
        frames = frames_for_date(
            client, collection, selection.resolved, root=root, write=write_metadata
        )
        plan.frames_by_day[selection.resolved] = frames
        for frame in frames:
            if frame.local_path(root, fmt).exists() and not force:
                plan.present.append(frame)
            else:
                plan.missing.append(frame)

    if probe and plan.missing:
        for frame in plan.missing:
            size = client.content_length(frame.url(fmt))
            if size is not None:
                plan.sizes[frame.image] = size
        plan.probed = True

    return plan


# ------------------------------------------------------------------- fetching


def _etag_store(root: Path) -> Path:
    return _cache_path(root, "etags.json")


def _load_etags(root: Path) -> dict[str, str]:
    path = _etag_store(root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_etags(root: Path, etags: dict[str, str]) -> None:
    path = _etag_store(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(etags, indent=1, sort_keys=True))


def fetch(
    client: EpicClient,
    plan: Plan,
    *,
    root: Path,
    max_bytes: int | None = None,
    revalidate: bool = False,
    force: bool = False,
    on_progress: Callable[[int, int, str, Frame], None] | None = None,
) -> FetchStats:
    """Download everything ``plan`` says is missing.

    The ceiling is checked twice. Up front against the projected total, so an
    over-budget run refuses before touching a body. Then before each file,
    against **bytes actually written so far** rather than the projection --
    which is what bounds the damage when the projection is wrong. A stream
    already in flight is never aborted, so the guarantee is: overshoot is at
    most one file.
    """
    stats = FetchStats()

    if max_bytes is not None and plan.mirror_bytes + plan.download_bytes > max_bytes:
        raise BudgetExceeded(
            f"{human_bytes(plan.mirror_bytes)} on disk plus "
            f"{human_bytes(plan.download_bytes)} to fetch "
            f"{'exceeds' if plan.probed else 'is projected to exceed'} the "
            f"{human_bytes(max_bytes)} ceiling"
        )

    targets = plan.missing if not revalidate else plan.missing + plan.present
    etags = _load_etags(root) if (revalidate or targets) else {}
    total = len(targets)

    for index, frame in enumerate(targets, start=1):
        target = frame.local_path(root, plan.fmt)
        key = str(target.relative_to(root))

        if target.exists() and not force and not revalidate:
            # Zero requests: the most polite outcome there is.
            stats.skipped += 1
            status = "skipped"
        elif max_bytes is not None and (
            # Reality first: what has actually landed already exhausts the cap.
            plan.mirror_bytes + stats.bytes_downloaded >= max_bytes
            # Then the projection, which stops us cleanly when it is accurate.
            or plan.mirror_bytes
            + stats.bytes_downloaded
            + plan.sizes.get(frame.image, NOMINAL_BYTES.get(plan.fmt, 0))
            > max_bytes
        ):
            stats.stopped_early = True
            if on_progress is not None:
                on_progress(index, total, "budget", frame)
            break
        else:
            try:
                status, written, etag = client.download(
                    frame.url(plan.fmt),
                    target,
                    etag=etags.get(key) if (revalidate and not force) else None,
                    resume=not force,
                )
            except EpicError as exc:
                stats.failed += 1
                stats.errors.append(f"{frame.image}: {exc}")
                status = "failed"
            else:
                if status == "current":
                    stats.current += 1
                else:
                    stats.downloaded += 1
                    stats.bytes_downloaded += written
                if etag:
                    etags[key] = etag

        if on_progress is not None:
            on_progress(index, total, status, frame)

    if etags:
        _save_etags(root, etags)
    return stats


# --------------------------------------------------------------------- helpers


_UNITS = (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10))


def human_bytes(count: int) -> str:
    for suffix, scale in _UNITS:
        if count >= scale:
            return f"{count / scale:.1f} {suffix}"
    return f"{count} B"


def parse_bytes(text: str) -> int:
    """Parse ``10GiB`` / ``500MB`` / ``1024`` into a byte count."""
    cleaned = text.strip().replace(" ", "").replace("_", "")
    scales = {
        "": 1,
        "b": 1,
        "k": 1 << 10, "kb": 1000, "kib": 1 << 10,
        "m": 1 << 20, "mb": 1000**2, "mib": 1 << 20,
        "g": 1 << 30, "gb": 1000**3, "gib": 1 << 30,
        "t": 1 << 40, "tb": 1000**4, "tib": 1 << 40,
    }
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-zA-Z]*)", cleaned)
    if not match:
        raise argparse.ArgumentTypeError(f"cannot parse a size from {text!r}")
    number, unit = match.groups()
    scale = scales.get(unit.lower())
    if scale is None:
        raise argparse.ArgumentTypeError(f"unknown size unit {unit!r} in {text!r}")
    return int(float(number) * scale)


def longitude_span(frames: Iterable[Frame]) -> float:
    """Total absolute longitude swept by a day's frames, in degrees.

    This sums the ``N-1`` gaps *between* frames, not the closing gap back to the
    first, so a full day reads ``360 * (N - 1) / N``: ~343 for 21 frames, ~332
    for 13. Verified against 2024-06-01, whose 22 frames sweep 157.4 -> -169.8
    monotonically. Well short of that means the day has a hole in its rotation.
    """
    lons = [f.lon for f in frames if f.lon is not None]
    if len(lons) < 2:
        return 0.0
    total = 0.0
    for previous, current in zip(lons, lons[1:]):
        step = (current - previous + 180.0) % 360.0 - 180.0
        total += abs(step)
    return total


# ------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixel-earth-fetch",
        description="Mirror DSCOVR/EPIC full-disc imagery from the NASA archive.",
        epilog=(
            "The archive publishes no rate limit; this client imposes its own. "
            "A full rotation of Earth is one UTC day of frames (9-22 depending "
            "on era)."
        ),
    )
    parser.add_argument(
        "--collection", choices=sorted(COLLECTION_PREFIXES), default="natural"
    )
    parser.add_argument("--format", choices=sorted(FORMAT_EXTENSIONS), default="png")
    parser.add_argument(
        "--root", type=Path, default=None, help="mirror root (default: <repo>/data/epic)"
    )

    picking = parser.add_argument_group("choosing days (a full rotation is one day)")
    picking.add_argument("--from", dest="start", type=date.fromisoformat, metavar="YYYY-MM-DD")
    picking.add_argument("--to", dest="end", type=date.fromisoformat, metavar="YYYY-MM-DD")
    picking.add_argument("--spread", type=int, metavar="N", help="N days evenly spaced in the range")
    picking.add_argument(
        "--date", dest="dates", action="append", type=date.fromisoformat, default=[],
        metavar="YYYY-MM-DD", help="explicit date, repeatable",
    )
    picking.add_argument("--last", type=int, metavar="N", help="the N most recent available days")

    politeness = parser.add_argument_group("politeness and budget")
    politeness.add_argument(
        "--max-bytes", type=parse_bytes, default="10GiB",
        help="refuse to exceed this total mirror size (default: %(default)s)",
    )
    politeness.add_argument(
        "--delay", type=float, default=0.5, metavar="SEC",
        help="minimum interval between requests (default: %(default)s)",
    )
    politeness.add_argument("--retries", type=int, default=5)
    politeness.add_argument("--user-agent", default=USER_AGENT)
    politeness.add_argument(
        "--no-probe", action="store_true",
        help="skip HEAD requests; project sizes from measured medians instead",
    )

    parser.add_argument(
        "--revalidate", action="store_true", help="conditional GET existing files (If-None-Match)"
    )
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    parser.add_argument("--dry-run", action="store_true", help="report the cost; write nothing")
    return parser


def _resolve_selections(client: EpicClient, args, root: Path) -> list[Selection]:
    available = available_dates(
        client, args.collection, root=root, write=not args.dry_run
    )
    if not available:
        return []

    if args.dates:
        return pick_explicit(available, args.dates)
    if args.last:
        return pick_last(available, args.last)
    if args.spread:
        start = args.start or available[0]
        end = args.end or available[-1]
        return pick_spread(available, start, end, args.spread)
    if args.start or args.end:
        start = args.start or available[0]
        end = args.end or available[-1]
        return [Selection(d, d) for d in available if start <= d <= end]
    return pick_last(available, 1)


def _print_plan(plan: Plan, root: Path, max_bytes: int | None) -> None:
    print(
        f"collection {plan.collection}, format {plan.fmt}"
        + (f", ceiling {human_bytes(max_bytes)}" if max_bytes else "")
    )
    print()
    for selection in plan.selections:
        frames = plan.frames_by_day.get(selection.resolved, [])
        missing = [f for f in frames if f in set(plan.missing)]
        size = sum(plan.sizes.get(f.image, NOMINAL_BYTES.get(plan.fmt, 0)) for f in missing)
        note = (
            f"  (requested {selection.requested}, snapped {selection.snap_days:+d}d)"
            if selection.snapped
            else ""
        )
        span = longitude_span(frames)
        print(
            f"  {selection.resolved}{note}"
            f"  {len(frames):>2} frames, {len(missing):>2} missing"
            f"  {human_bytes(size):>9}"
            f"  {span:5.0f}deg swept"
        )

    projected = plan.mirror_bytes + plan.download_bytes
    print()
    print(
        f"  {plan.total_frames} frames, {len(plan.present)} already present, "
        f"{human_bytes(plan.download_bytes)} to download"
        + ("" if plan.probed else " (projected, no HEAD probe)")
    )
    share = f"  ({projected / max_bytes:.1%} of ceiling)" if max_bytes else ""
    print(f"  mirror {human_bytes(plan.mirror_bytes)} -> {human_bytes(projected)}{share}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from pixel_earth.batch import find_repo_root

    root = args.root or (find_repo_root() / "data" / "epic")

    with EpicClient(
        delay=args.delay, retries=args.retries, user_agent=args.user_agent
    ) as client:
        try:
            selections = _resolve_selections(client, args, root)
            if not selections:
                print("no available dates matched that selection")
                return 1

            plan = build_plan(
                client,
                root=root,
                collection=args.collection,
                fmt=args.format,
                selections=selections,
                probe=not args.no_probe,
                write_metadata=not args.dry_run,
                force=args.force,
            )
            _print_plan(plan, root, args.max_bytes)

            if args.dry_run:
                print("\ndry run: nothing written")
                return 0
            if not plan.missing and not args.revalidate:
                print("\nnothing to do")
                return 0

            print()

            def progress(index: int, total: int, status: str, frame: Frame) -> None:
                print(f"[{index}/{total}] {status:<10} {frame.image}")

            stats = fetch(
                client,
                plan,
                root=root,
                max_bytes=args.max_bytes,
                revalidate=args.revalidate,
                force=args.force,
                on_progress=progress,
            )
        except BudgetExceeded as exc:
            print(f"refusing to start: {exc}")
            return 2
        except EpicError as exc:
            print(f"error: {exc}")
            return 2

    print(
        f"\ndownloaded {stats.downloaded} ({human_bytes(stats.bytes_downloaded)})"
        f"  skipped {stats.skipped}  unchanged {stats.current}  failed {stats.failed}"
        f"  [{client.request_count} requests]"
    )
    if stats.stopped_early:
        print("stopped early: byte ceiling reached")
    for message in stats.errors[:10]:
        print(f"  {message}")
    print(f"mirror: {root}")

    return 1 if (stats.failed or stats.stopped_early) else 0


if __name__ == "__main__":
    raise SystemExit(main())
