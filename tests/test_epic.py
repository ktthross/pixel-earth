"""Hermetic tests for the EPIC fetcher.

Every HTTP interaction goes through ``httpx.MockTransport`` and every wait
through a fake clock, so the suite never touches NASA and never sleeps. The
politeness guarantees get the most attention here, precisely because the server
enforces none of them and so cannot tell us when we get them wrong.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest

from pixel_earth.epic import (
    BASE_URL,
    NOMINAL_BYTES,
    BudgetExceeded,
    EpicClient,
    EpicError,
    Frame,
    Plan,
    RateLimiter,
    Selection,
    available_dates,
    build_plan,
    fetch,
    frames_for_date,
    human_bytes,
    longitude_span,
    mirror_bytes,
    parse_bytes,
    _load_etags,
    _save_etags,
    parse_frames,
    pick_explicit,
    pick_last,
    pick_spread,
)


# ------------------------------------------------------------------- fixtures


class FakeClock:
    """Monotonic clock whose only way to advance is sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Recorder:
    """MockTransport handler that logs requests and replays scripted responses."""

    def __init__(self, routes: dict[str, list[httpx.Response]] | None = None) -> None:
        self.routes = routes or {}
        self.requests: list[httpx.Request] = []
        self.default = httpx.Response(404)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queued = self.routes.get(str(request.url))
        if not queued:
            return self.default
        return queued.pop(0) if len(queued) > 1 else queued[0]

    @property
    def methods(self) -> list[str]:
        return [r.method for r in self.requests]

    def header_values(self, name: str) -> list[str | None]:
        return [r.headers.get(name) for r in self.requests]


def make_client(recorder: Recorder, clock: FakeClock | None = None, **kwargs) -> EpicClient:
    clock = clock or FakeClock()
    return EpicClient(
        client=httpx.Client(transport=httpx.MockTransport(recorder)),
        clock=clock.time,
        sleep=clock.sleep,
        **kwargs,
    )


def frame_json(stamp: str, *, lon: float = 0.0, lat: float = 20.0, prefix: str = "epic_1b_") -> dict:
    return {
        "identifier": stamp,
        "image": f"{prefix}{stamp}",
        "version": "03",
        "caption": "This image was taken by the DSCOVR EPIC camera",
        "date": f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}",
        "centroid_coordinates": {"lat": lat, "lon": lon},
        "dscovr_j2000_position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "lunar_j2000_position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "sun_j2000_position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "attitude_quaternions": {"q0": 0.0, "q1": 0.0, "q2": 0.0, "q3": 1.0},
        "coords": {},
    }


A_FRAME = Frame(
    collection="natural",
    image="epic_1b_20240601004554",
    archive_day=date(2024, 6, 1),
    captured=None,
    lat=20.6,
    lon=157.4,
)

A_URL = f"{BASE_URL}/archive/natural/2024/06/01/png/epic_1b_20240601004554.png"


# ------------------------------------------------------------- URLs and paths


def test_archive_url_matches_the_documented_schema():
    assert A_FRAME.url("png") == (
        "https://epic.gsfc.nasa.gov/archive/natural/2024/06/01/png/"
        "epic_1b_20240601004554.png"
    )
    # thumbs holds jpgs, not files named .thumbs
    assert A_FRAME.url("thumbs").endswith("/thumbs/epic_1b_20240601004554.jpg")
    assert A_FRAME.url("jpg").endswith("/jpg/epic_1b_20240601004554.jpg")


def test_url_zero_pads_month_and_day():
    frame = Frame("natural", "epic_1b_20150703010203", date(2015, 7, 3), None, 0.0, 0.0)

    assert "/2015/07/03/" in frame.url("png")


def test_url_uses_the_per_collection_prefix():
    enhanced = parse_frames([frame_json("20240601004554", prefix="epic_RGB_")], "enhanced")[0]

    assert enhanced.url("png") == (
        "https://epic.gsfc.nasa.gov/archive/enhanced/2024/06/01/png/"
        "epic_RGB_20240601004554.png"
    )


def test_local_path_mirrors_the_archive_path(tmp_path):
    relative = A_FRAME.local_path(tmp_path, "png").relative_to(tmp_path)

    assert relative == Path("natural/2024/06/01/png/epic_1b_20240601004554.png")


# ------------------------------------------------------------ metadata parsing


def test_parse_frames_reads_the_real_field_set():
    frames = parse_frames([frame_json("20240601004554", lon=157.4, lat=20.6)], "natural")

    assert len(frames) == 1
    assert frames[0].image == "epic_1b_20240601004554"
    assert (frames[0].lat, frames[0].lon) == (20.6, 157.4)


def test_archive_day_comes_from_the_image_name_not_the_date_field():
    """They differ: on 2024-06-01 the name says 00:45:54 and date says 00:41:06.

    Only the name's date appears in the archive URL, so that is what we use.
    """
    entry = frame_json("20240601004554")
    entry["date"] = "2024-06-01 00:41:06"
    frame = parse_frames([entry], "natural")[0]

    assert frame.archive_day == date(2024, 6, 1)
    assert frame.captured.minute == 41  # metadata instant preserved separately
    assert "20240601004554" in frame.url("png")  # URL keyed off the name


def test_parse_frames_skips_unplaceable_entries():
    assert parse_frames([{"image": "not_a_timestamp"}, {}], "natural") == []


def test_empty_response_means_no_imagery_not_an_error(tmp_path):
    """The API returns [] for the archive's gaps rather than a 404."""
    recorder = Recorder(
        {f"{BASE_URL}/api/natural/date/2019-09-15": [httpx.Response(200, json=[])]}
    )
    with make_client(recorder) as client:
        frames = frames_for_date(client, "natural", date(2019, 9, 15), root=tmp_path)

    assert frames == []


def test_metadata_is_cached_beside_the_images(tmp_path):
    url = f"{BASE_URL}/api/natural/date/2024-06-01"
    recorder = Recorder({url: [httpx.Response(200, json=[frame_json("20240601004554")])]})
    with make_client(recorder) as client:
        first = frames_for_date(client, "natural", date(2024, 6, 1), root=tmp_path)
        second = frames_for_date(client, "natural", date(2024, 6, 1), root=tmp_path)

    assert len(first) == len(second) == 1
    # A past day is immutable, so the second call must not hit the network.
    assert len(recorder.requests) == 1
    assert (tmp_path / "natural/2024/06/01/metadata.json").exists()


def test_available_dates_parses_and_caches(tmp_path):
    url = f"{BASE_URL}/api/natural/available"
    recorder = Recorder({url: [httpx.Response(200, json=["2015-06-13", "2024-06-01"])]})
    with make_client(recorder) as client:
        first = available_dates(client, "natural", root=tmp_path)
        second = available_dates(client, "natural", root=tmp_path)

    assert first == [date(2015, 6, 13), date(2024, 6, 1)] == second
    assert len(recorder.requests) == 1


def test_available_dates_rejects_a_non_list_response(tmp_path):
    recorder = Recorder(
        {f"{BASE_URL}/api/natural/available": [httpx.Response(200, json={"oops": 1})]}
    )
    with make_client(recorder) as client, pytest.raises(EpicError):
        available_dates(client, "natural", root=tmp_path)


# --------------------------------------------------------------- date picking


def test_pick_spread_spaces_evenly_and_snaps_to_available():
    available = [date(2024, 1, 1) + timedelta(days=i) for i in range(366)]
    picks = pick_spread(available, date(2024, 1, 1), date(2024, 12, 31), 6)

    assert len(picks) == 6
    assert [p.resolved for p in picks] == sorted(p.resolved for p in picks)
    gaps = [
        (b.resolved - a.resolved).days for a, b in zip(picks, picks[1:])
    ]
    assert max(gaps) - min(gaps) <= 1  # evenly spaced
    assert all(not p.snapped for p in picks)  # every target existed


def test_pick_spread_reports_how_far_it_snapped_across_a_gap():
    """The real 2019-06-27 -> 2020-02-12 safe-hold hole, in miniature."""
    available = [date(2019, 6, 20) + timedelta(days=i) for i in range(8)]
    available += [date(2020, 2, 12) + timedelta(days=i) for i in range(8)]

    picks = pick_spread(available, date(2019, 6, 20), date(2020, 2, 19), 3)

    assert len(picks) == 3
    assert all(p.resolved in available for p in picks)
    # The midpoint target lands inside the hole and must move to reach imagery.
    assert any(p.snapped for p in picks)
    assert any(abs(p.snap_days) > 30 for p in picks)


def test_pick_spread_returns_distinct_dates_when_targets_share_a_neighbour():
    """Three clustered dates, targets spread across a wide range.

    Picking nearest-available independently would map both the middle and the
    late target onto 01-11; the greedy pass must hand out distinct dates.
    """
    available = [date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)]
    picks = pick_spread(available, date(2024, 1, 1), date(2024, 1, 20), 3)

    assert len({p.resolved for p in picks}) == 3
    assert [p.resolved for p in picks] == available


def test_pick_spread_returns_everything_when_asked_for_more_than_exists():
    available = [date(2024, 1, 1), date(2024, 1, 5)]

    assert len(pick_spread(available, date(2024, 1, 1), date(2024, 1, 5), 10)) == 2


def test_pick_spread_handles_empty_and_zero():
    assert pick_spread([], date(2024, 1, 1), date(2024, 1, 5), 3) == []
    assert pick_spread([date(2024, 1, 1)], date(2024, 1, 1), date(2024, 1, 5), 0) == []


def test_pick_spread_of_one_takes_the_middle():
    available = [date(2024, 1, 1) + timedelta(days=i) for i in range(11)]
    picks = pick_spread(available, date(2024, 1, 1), date(2024, 1, 11), 1)

    assert picks[0].resolved == date(2024, 1, 6)


def test_pick_last_and_explicit():
    available = [date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 9)]

    assert [p.resolved for p in pick_last(available, 2)] == [date(2024, 1, 5), date(2024, 1, 9)]
    # An unavailable request snaps to the nearest date that exists.
    snapped = pick_explicit(available, [date(2024, 1, 6)])[0]
    assert snapped.resolved == date(2024, 1, 5)
    assert snapped.snap_days == -1


# -------------------------------------------------------------------- planning


def test_build_plan_splits_present_from_missing_and_probes_sizes(tmp_path):
    meta = f"{BASE_URL}/api/natural/date/2024-06-01"
    frames = [frame_json("20240601004554"), frame_json("20240601014554")]
    present = A_FRAME.local_path(tmp_path, "png")
    present.parent.mkdir(parents=True)
    present.write_bytes(b"x" * 100)

    recorder = Recorder(
        {
            meta: [httpx.Response(200, json=frames)],
            f"{BASE_URL}/archive/natural/2024/06/01/png/epic_1b_20240601014554.png": [
                httpx.Response(200, headers={"Content-Length": "2779794"})
            ],
        }
    )
    with make_client(recorder) as client:
        plan = build_plan(
            client,
            root=tmp_path,
            collection="natural",
            fmt="png",
            selections=[Selection(date(2024, 6, 1), date(2024, 6, 1))],
        )

    assert len(plan.present) == 1 and len(plan.missing) == 1
    assert plan.probed
    assert plan.download_bytes == 2779794  # exact, only the missing one
    # Only the missing file was HEADed; the present one cost nothing.
    assert recorder.methods.count("HEAD") == 1


def test_build_plan_projects_size_when_probing_is_skipped(tmp_path):
    meta = f"{BASE_URL}/api/natural/date/2024-06-01"
    recorder = Recorder({meta: [httpx.Response(200, json=[frame_json("20240601004554")])]})
    with make_client(recorder) as client:
        plan = build_plan(
            client,
            root=tmp_path,
            collection="natural",
            fmt="png",
            selections=[Selection(date(2024, 6, 1), date(2024, 6, 1))],
            probe=False,
        )

    assert not plan.probed
    assert plan.download_bytes == NOMINAL_BYTES["png"]
    assert "HEAD" not in recorder.methods


def test_mirror_bytes_ignores_partial_downloads(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x" * 10)
    (tmp_path / "b.png.part").write_bytes(b"x" * 999)

    assert mirror_bytes(tmp_path) == 10


# ------------------------------------------------ politeness: zero-cost skips


def test_an_existing_file_costs_zero_requests(tmp_path):
    target = A_FRAME.local_path(tmp_path, "png")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"complete")

    recorder = Recorder()
    plan = Plan(collection="natural", fmt="png", selections=[], missing=[A_FRAME])
    with make_client(recorder) as client:
        stats = fetch(client, plan, root=tmp_path)

    assert stats.skipped == 1 and stats.downloaded == 0
    assert recorder.requests == []  # not even a HEAD


def test_revalidate_sends_if_none_match_and_accepts_304(tmp_path):

    target = A_FRAME.local_path(tmp_path, "png")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"complete")
    _save_etags(tmp_path, {str(target.relative_to(tmp_path)): '"abc123"'})

    recorder = Recorder({A_URL: [httpx.Response(304)]})
    plan = Plan(collection="natural", fmt="png", selections=[], present=[A_FRAME])
    with make_client(recorder) as client:
        stats = fetch(client, plan, root=tmp_path, revalidate=True)

    assert stats.current == 1 and stats.downloaded == 0
    assert recorder.header_values("If-None-Match") == ['"abc123"']
    assert target.read_bytes() == b"complete"  # untouched


# ----------------------------------------------- politeness: integrity/resume


def test_download_is_atomic_and_stores_the_etag(tmp_path):

    body = b"y" * 2048
    recorder = Recorder(
        {A_URL: [httpx.Response(200, content=body, headers={"ETag": '"deadbeef"'})]}
    )
    plan = Plan(collection="natural", fmt="png", selections=[], missing=[A_FRAME])
    with make_client(recorder) as client:
        stats = fetch(client, plan, root=tmp_path)

    target = A_FRAME.local_path(tmp_path, "png")
    assert stats.downloaded == 1 and stats.bytes_downloaded == 2048
    assert target.read_bytes() == body
    assert not target.with_name(target.name + ".part").exists()
    assert _load_etags(tmp_path)[str(target.relative_to(tmp_path))] == '"deadbeef"'


def test_a_partial_file_resumes_with_a_range_request(tmp_path):

    target = A_FRAME.local_path(tmp_path, "png")
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"a" * 500)

    recorder = Recorder(
        {
            A_URL: [
                httpx.Response(
                    206,
                    content=b"b" * 1500,
                    headers={
                        "Content-Length": "1500",
                        "Content-Range": "bytes 500-1999/2000",
                    },
                )
            ]
        }
    )
    plan = Plan(collection="natural", fmt="png", selections=[], missing=[A_FRAME])
    with make_client(recorder) as client:
        stats = fetch(client, plan, root=tmp_path)

    assert recorder.header_values("Range") == ["bytes=500-"]
    assert target.read_bytes() == b"a" * 500 + b"b" * 1500
    assert stats.bytes_downloaded == 1500  # only the new bytes counted
    assert not part.exists()


def test_a_truncated_body_never_lands_at_the_final_path(tmp_path):

    # Declares 2000 bytes, sends 10.
    recorder = Recorder(
        {A_URL: [httpx.Response(200, content=b"z" * 10, headers={"Content-Length": "2000"})]}
    )
    plan = Plan(collection="natural", fmt="png", selections=[], missing=[A_FRAME])
    with make_client(recorder, retries=2) as client:
        stats = fetch(client, plan, root=tmp_path)

    target = A_FRAME.local_path(tmp_path, "png")
    assert stats.failed == 1 and stats.downloaded == 0
    assert not target.exists()
    # The .part survives so the next run can resume rather than restart.
    assert target.with_name(target.name + ".part").exists()


def test_force_ignores_both_the_local_file_and_the_partial(tmp_path):

    target = A_FRAME.local_path(tmp_path, "png")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")
    target.with_name(target.name + ".part").write_bytes(b"junk" * 10)

    recorder = Recorder({A_URL: [httpx.Response(200, content=b"fresh")]})
    plan = Plan(collection="natural", fmt="png", selections=[], missing=[A_FRAME])
    with make_client(recorder) as client:
        fetch(client, plan, root=tmp_path, force=True)

    assert target.read_bytes() == b"fresh"
    assert recorder.header_values("Range") == [None]  # no resume when forcing


# ------------------------------------------------- politeness: retry / backoff


def test_a_503_is_retried_and_then_succeeds(tmp_path):

    clock = FakeClock()
    recorder = Recorder(
        {A_URL: [httpx.Response(503), httpx.Response(503), httpx.Response(200, content=b"ok")]}
    )
    plan = Plan(collection="natural", fmt="png", selections=[], missing=[A_FRAME])
    with make_client(recorder, clock, retries=5, backoff=1.0) as client:
        stats = fetch(client, plan, root=tmp_path)

    assert stats.downloaded == 1
    assert len(recorder.requests) == 3
    # Exponential: first wait 1s, second 2s.
    assert [s for s in clock.slept if s >= 1.0] == [1.0, 2.0]


def test_retry_after_is_honoured_over_the_backoff_schedule(tmp_path):

    clock = FakeClock()
    recorder = Recorder(
        {
            A_URL: [
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, content=b"ok"),
            ]
        }
    )
    plan = Plan(collection="natural", fmt="png", selections=[], missing=[A_FRAME])
    with make_client(recorder, clock, backoff=1.0) as client:
        fetch(client, plan, root=tmp_path)

    assert 7.0 in clock.slept


def test_giving_up_records_a_failure_rather_than_raising(tmp_path):

    recorder = Recorder({A_URL: [httpx.Response(503)]})
    plan = Plan(collection="natural", fmt="png", selections=[], missing=[A_FRAME])
    with make_client(recorder, retries=3) as client:
        stats = fetch(client, plan, root=tmp_path)

    assert stats.failed == 1 and stats.errors
    assert len(recorder.requests) == 3


def test_a_404_is_not_retried(tmp_path):
    recorder = Recorder({A_URL: [httpx.Response(404)]})
    with make_client(recorder, retries=5) as client:
        with pytest.raises(EpicError):
            client.download(A_URL, tmp_path / "missing.png")

    assert len(recorder.requests) == 1  # a missing file is not a transient fault


# ------------------------------------------------------ politeness: rate limit


def test_rate_limiter_enforces_the_minimum_interval():
    clock = FakeClock()
    limiter = RateLimiter(0.5, clock=clock.time, sleep=clock.sleep)

    limiter.wait()  # first call is free
    limiter.wait()
    limiter.wait()

    assert clock.slept == [0.5, 0.5]


def test_rate_limiter_does_not_wait_when_time_already_passed():
    clock = FakeClock()
    limiter = RateLimiter(0.5, clock=clock.time, sleep=clock.sleep)

    limiter.wait()
    clock.now += 10.0
    limiter.wait()

    assert clock.slept == []


def test_every_request_goes_through_the_limiter(tmp_path):

    clock = FakeClock()
    frames = [
        Frame("natural", f"epic_1b_2024060100{i:02d}00", date(2024, 6, 1), None, 0.0, 0.0)
        for i in range(3)
    ]
    recorder = Recorder({f.url("png"): [httpx.Response(200, content=b"x")] for f in frames})
    plan = Plan(collection="natural", fmt="png", selections=[], missing=frames)
    with make_client(recorder, clock, delay=0.5) as client:
        fetch(client, plan, root=tmp_path)

    assert len(recorder.requests) == 3
    assert clock.slept == [0.5, 0.5]  # gated between each


# ------------------------------------------------------- politeness: byte cap


def test_the_ceiling_refuses_before_any_body_is_requested(tmp_path):

    recorder = Recorder()
    plan = Plan(
        collection="natural",
        fmt="png",
        selections=[],
        missing=[A_FRAME],
        sizes={A_FRAME.image: 5_000_000},
        probed=True,
    )
    with make_client(recorder) as client:
        with pytest.raises(BudgetExceeded, match="ceiling"):
            fetch(client, plan, root=tmp_path, max_bytes=1_000_000)

    assert recorder.requests == []


def test_the_ceiling_bounds_the_overshoot_when_the_projection_was_wrong(tmp_path):
    """A lying projection can overshoot by at most one file, not unboundedly.

    A stream already in flight is not aborted, so one file's worth is the honest
    guarantee. Without the reality check, 40x-too-small sizes would let the whole
    set through.
    """
    frames = [
        Frame("natural", f"epic_1b_2024060100{i:02d}00", date(2024, 6, 1), None, 0.0, 0.0)
        for i in range(4)
    ]
    recorder = Recorder(
        {f.url("png"): [httpx.Response(200, content=b"x" * 400)] for f in frames}
    )
    # Claims 10 bytes each; each body is actually 400.
    plan = Plan(
        collection="natural",
        fmt="png",
        selections=[],
        missing=frames,
        sizes={f.image: 10 for f in frames},
        probed=True,
    )
    with make_client(recorder) as client:
        stats = fetch(client, plan, root=tmp_path, max_bytes=1000)

    assert stats.stopped_early
    assert stats.downloaded < 4
    # Overshoot bounded by the single in-flight file (400 B), not by the 1600 B
    # the whole set would have cost.
    assert stats.bytes_downloaded <= 1000 + 400


def test_mirror_size_already_on_disk_counts_against_the_ceiling(tmp_path):

    plan = Plan(
        collection="natural",
        fmt="png",
        selections=[],
        missing=[A_FRAME],
        sizes={A_FRAME.image: 100},
        mirror_bytes=990,
        probed=True,
    )
    with make_client(Recorder()) as client:
        with pytest.raises(BudgetExceeded):
            fetch(client, plan, root=tmp_path, max_bytes=1000)


# ------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1024", 1024),
        ("10GiB", 10 * 1024**3),
        ("500MiB", 500 * 1024**2),
        ("500MB", 500 * 1000**2),
        ("1.5GiB", int(1.5 * 1024**3)),
        ("2g", 2 * 1024**3),
    ],
)
def test_parse_bytes(text, expected):
    assert parse_bytes(text) == expected


def test_parse_bytes_rejects_nonsense():
    import argparse

    for bad in ("banana", "10 furlongs", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_bytes(bad)


def test_human_bytes():
    assert human_bytes(2_779_794) == "2.7 MiB"
    assert human_bytes(500) == "500 B"


def test_longitude_span_measures_a_full_rotation():
    """22 frames at ~16 degrees apart is one complete rotation."""
    frames = [
        Frame("natural", f"epic_1b_2024060100{i:02d}00", date(2024, 6, 1), None, 20.0,
              (157.4 - 16.36 * i + 180) % 360 - 180)
        for i in range(22)
    ]

    assert longitude_span(frames) == pytest.approx(343.6, abs=1.0)  # 21 steps of 16.36


def test_longitude_span_of_too_few_frames_is_zero():
    assert longitude_span([]) == 0.0
    assert longitude_span([A_FRAME]) == 0.0


# ----------------------------------------------------------------- live check


@pytest.mark.network
def test_live_thumb_download_still_works(tmp_path):
    """Opt-in (`pytest -m network`): catches upstream drift. ~6 KiB."""
    frame = Frame(
        "natural", "epic_1b_20240601004554", date(2024, 6, 1), None, 20.6, 157.4
    )
    with EpicClient(delay=0.0) as client:
        status, written, etag = client.download(frame.url("thumbs"), frame.local_path(tmp_path, "thumbs"))

    assert status == "downloaded"
    assert 1000 < written < 100_000
    assert frame.local_path(tmp_path, "thumbs").exists()
