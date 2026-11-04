#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "numpy"]
# ///
"""Yandex Panoramas (the main street view for Russia, Belarus, Kazakhstan, Türkiye and neighbours): find points,
see dates and nearby points, render by heading, build comparison sheets. Usage mirrors gsv.py.

Uses the endpoints the Yandex Maps web page itself calls; no key needed.
Coverage: https://yandex.com/maps/streetview/ — mostly RU, BY, KZ, TR, AM, GE, UZ. Outside this area use gsv.py.

What it has, and how it differs from the other two panorama scripts:
- Coverage is the point: Yandex drives almost every Russian road and keeps 3–5 historical captures per point in
  cities (2010–today), where Google Street View is sparse and years old; in China use baidu_pano.py as before.
- Rendering works differently: Google and Baidu serve a finished perspective image per request, Yandex serves only
  equirectangular tiles — `render`/`sheet` stitch them locally and project the view themselves. Consequences:
  no logo or watermark is baked into the image (LOGO_BOX is None for match.py), resolution is chosen with `--zoom`,
  and the first view of a panorama downloads its tiles (~80 small requests, reused afterwards from .geo-cache).
- The panoid encodes the capture date (last numeric segment is a unix timestamp), so `near`/`info` return the date
  and historical captures without extra requests; `sheet --date 2018` renders only that year's capture.
- `scan` matches baidu_pano.py's scan output ({id: {wgs, road, date}}), so `match.py rank --panos` works unchanged,
  and `match.py --render yandex` is picked automatically from the panoid format (digits_digits_digits_digits).
- Not implemented yet (baidu_pano.py/gsv.py have them): `sample` (random city fixture sampling) and `targets`
  (panoramas around each target point); use gsv.py targets where Google has coverage.

Examples:
  yandex_pano.py near 55.7558,37.6176 --radius 50              # nearest panorama point: id, coordinates, capture date, historical captures, street, nearby points
  yandex_pano.py render 1298139818_673202752_23_1758020864 --heading 0 --out v.jpg
  yandex_pano.py sheet --at 55.7558,37.6176 --headings 0,60,120,180,240,300 --out around.jpg   # look around from a single point
  yandex_pano.py sheet --ids ID1,ID2 --toward 55.7560,37.6180 --out s.jpg                       # every point faces the same target
  yandex_pano.py scan 55.7558,37.6176 --radius 200 --out panos.json                            # every point in the area, for match.py rank --panos
"""
from __future__ import annotations

import argparse
from _net import curl_args, PROXY_HELP
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import geo  # noqa: E402
from baidu_pano import _font  # noqa: E402

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
API = "https://api-maps.yandex.ru/services/panoramas/1.x/"
TILE = "https://pano.maps.yandex.net/{image_id}/{z}.{x}.{y}"


class ServiceError(RuntimeError):
    """Yandex answered with an error instead of a lookup result (endpoint changed, blocked, rate-limited)."""


def _curl(url: str, proxy: str | None, timeout: int = 30) -> bytes:
    cmd = ["curl", "-q", "-fsS", "-m", str(timeout), "-A", UA]
    cmd += curl_args(proxy)
    for attempt in range(2):
        r = subprocess.run(cmd + [url], capture_output=True)
        if not r.returncode:
            return r.stdout
    raise ServiceError(f"request failed (curl exit {r.returncode}); run doctor.py --network to check connectivity")


def _pano_date(oid: str) -> str | None:
    """Capture month: the panoid's last numeric segment is a unix timestamp."""
    for seg in reversed(oid.split("_")):
        if seg.isdigit() and len(seg) >= 10:
            return datetime.fromtimestamp(int(seg), timezone.utc).strftime("%Y-%m")
    return None


def _oid_from_url(url: str) -> str | None:
    m = re.search(r"oid=([^&]+)", url)
    return m.group(1) if m else None


def _parse(raw: dict) -> dict | None:
    data = raw.get("data")
    if not data or not data.get("Data"):
        return None
    d = data["Data"]
    ann = data.get("Annotation") or {}
    oid = str(d["panoramaId"])
    coords = d["Point"]["coordinates"]
    origin = d["EquirectangularProjection"]["Origin"]
    zooms = {int(z["level"]): [int(z["width"]), int(z["height"])] for z in d["Images"]["Zooms"]}
    hist = []
    for h in ann.get("HistoricalPanoramas") or []:
        try:
            c = h["Connection"]
            hid = str(c.get("oid") or _oid_from_url(c.get("href", "")) or "")
            if hid and hid != oid:
                hist.append({"id": hid, "wgs": [c["Point"]["coordinates"][1], c["Point"]["coordinates"][0]],
                             "date": _pano_date(hid)})
        except (KeyError, TypeError, IndexError, ValueError):
            continue
    hist.sort(key=lambda p: p.get("date") or "", reverse=True)
    nbrs = []
    for n in (ann.get("Graph") or {}).get("Nodes") or []:
        if n.get("panoid") and str(n["panoid"]) != oid:
            nbrs.append({"id": str(n["panoid"]), "wgs": [float(n["lat"]), float(n["lon"])]})
    links = []
    for t in ann.get("Thoroughfares") or []:
        hid = _oid_from_url(t.get("Connection", {}).get("href", ""))
        if hid:
            links.append({"id": hid, "dir": float(t["Direction"][0])})  # compass bearing (0=N, clockwise)
    addr = []
    for m in ann.get("Markers") or []:
        try:
            if m["geometry"]["coordinates"][2] == 7:  # address markers sit at 7 m, the rest at 2 m
                addr.append(m["properties"].get("description") or m["properties"].get("name", ""))
        except (KeyError, TypeError, IndexError):
            continue
    date = _pano_date(oid)
    return {"id": oid, "wgs": [float(coords[1]), float(coords[0])], "height": float(coords[2]) if len(coords) > 2 else None,
            "street": d["Point"].get("name", ""), "address": " / ".join(a for a in addr if a),
            "date": date, "dates_seen": sorted({p["date"] for p in ([{"date": date}] + hist) if p.get("date")}),
            "heading0": float(origin[0]), "pitch0": float(origin[1]),
            "image_id": d["Images"]["imageId"], "tiles": [int(d["Images"]["Tiles"]["width"]), int(d["Images"]["Tiles"]["height"])],
            "zooms": zooms, "history": hist, "neighbors": nbrs[:60], "links": links}


def near(lat: float, lon: float, radius: float = 50, proxy: str | None = None) -> dict | None:
    """Nearest panorama within radius metres; None when there is none. Raises ServiceError when the lookup itself fails."""
    t = _curl(f"{API}?l=stv&lang=ru_RU&ll={lon:.6f},{lat:.6f}&origin=userAction&provider=streetview", proxy)
    try:
        raw = json.loads(t)
    except json.JSONDecodeError:
        raise ServiceError(f"unreadable reply: {' '.join(t.decode('utf-8', 'replace').split())[:160] or 'empty (network?)'}") from None
    if raw.get("status") != "success":
        return None
    p = _parse(raw)
    if p is None:
        return None
    return p if geo.distance((lat, lon), tuple(p["wgs"])) <= radius else None


def info(oid: str, proxy: str | None = None, cache: Path | None = None) -> dict | None:
    """Panorama metadata by id; cached under cache/meta/ when a cache dir is given (match.py asks for every candidate)."""
    p = (cache / "meta" / f"{oid}.json") if cache else None
    if p and p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            p.unlink(missing_ok=True)
    t = _curl(f"{API}?l=stv&lang=ru_RU&oid={oid}&origin=userAction&provider=streetview", proxy)
    try:
        raw = json.loads(t)
    except json.JSONDecodeError:
        raise ServiceError(f"unreadable reply: {' '.join(t.decode('utf-8', 'replace').split())[:160] or 'empty (network?)'}") from None
    res = _parse(raw) if raw.get("status") == "success" else None
    if res and p:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    return res


def pick_date(res: dict, date: str) -> dict | None:
    """Pick one capture ('2018' or '2018-07') from a near() result: this panorama and its historical captures."""
    for p in [{"id": res["id"], "wgs": res["wgs"], "date": res["date"]}, *res.get("history", [])]:
        if p.get("date") and p["date"].startswith(date):
            return p
    return None


# --- tiles and perspective rendering -------------------------------------------------------------------------------

def _tile(image_id: str, z: int, x: int, y: int, proxy: str | None, cache: Path) -> Image.Image:
    """One panorama tile; fetched at most once and kept in the cache (a missing tile stays black)."""
    p = cache / "tiles" / image_id / f"{z}.{x}.{y}.jpg"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = _curl(TILE.format(image_id=image_id, z=z, x=x, y=y), proxy)
        except ServiceError:
            return Image.new("RGB", (256, 256), "black")  # a failed fetch is not cached, the next run retries it
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
    try:
        return Image.open(p).convert("RGB")
    except Exception:  # noqa: BLE001
        return Image.new("RGB", (256, 256), "black")


def stitch(p: dict, zoom: int, proxy: str | None, cache: Path) -> Image.Image:
    """Equirectangular panorama at the given zoom level (0 = largest), built from 256px tiles."""
    zoom = max(0, min(zoom, max(p["zooms"])))
    W, H = p["zooms"][zoom]
    tw, th = p["tiles"]
    out = cache / "full" / f"{p['image_id']}_z{zoom}.jpg"
    if out.exists():
        try:
            im = Image.open(out)
            if im.size == (W, H):
                return im.convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    cols, rows = math.ceil(W / tw), math.ceil(H / th)
    im = Image.new("RGB", (W, H), "black")
    with ThreadPoolExecutor(12) as ex:
        done = list(ex.map(lambda a: (a, _tile(p["image_id"], zoom, *a, proxy, cache)),
                           [(x, y) for x in range(cols) for y in range(rows)]))
    for (x, y), t in done:
        im.paste(t, (x * tw, y * th))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    im.save(tmp, format="JPEG", quality=90)
    os.replace(tmp, out)
    return im


def render(p: dict, heading: float, pitch: float = 0, fov: float = 90,
           w: int = 1024, h: int = 768, zoom: int = 2,
           proxy: str | None = None, cache: Path | None = None) -> Image.Image:
    """Perspective view from the stitched equirect. heading: compass bearing (0=N, clockwise);
    positive pitch = looking up; fov: horizontal field of view. No logo or watermark is baked into the image."""
    cache = cache or Path(".geo-cache/yandex")
    key = f"{p['id']}_{heading:.0f}_{pitch:.0f}_{fov:.0f}_{w}x{h}.jpg"
    out = cache / "views" / key
    if out.exists() and out.stat().st_size > 2000:
        try:
            return Image.open(out).convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    eq = np.asarray(stitch(p, zoom, proxy, cache))
    H_, W_ = eq.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    s = math.tan(math.radians(fov) / 2) / (w / 2)
    x_cam = (xs - w / 2) * s
    y_cam = -(ys - h / 2) * s
    z_cam = np.ones_like(x_cam)
    az = np.arctan2(x_cam, z_cam)                                   # + to the right of the view axis
    el = pitch + np.degrees(np.arctan2(y_cam, np.sqrt(x_cam**2 + z_cam**2)))
    bearing = (heading + np.degrees(az)) % 360.0
    # The stitched image is a standard equirect: x = 0 is 180° left of the projection origin (compass),
    # the top edge is the zenith and the bottom part is cropped off (H/W < 0.5).
    yaw0 = (p["heading0"] - 180.0) % 360.0
    vfov_img = H_ / W_ * 360.0
    px = (bearing - yaw0) % 360.0 / 360.0 * W_
    py = (90.0 - el) / vfov_img * H_
    px = np.clip(px, 0, W_ - 1).astype(np.int32)
    py = np.clip(py, 0, H_ - 1).astype(np.int32)
    im = Image.fromarray(eq[py, px])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    im.save(tmp, format="JPEG", quality=92)
    os.replace(tmp, out)
    return im


# No logo or watermark: views come from our own stitch of the raw tiles, nothing is baked in.
LOGO_BOX = None


def sheet(items: list[dict], out: Path, proxy: str | None, cache: Path, cols: int = 3, tw: int = 480, th: int = 360) -> None:
    """items: [{meta, heading, pitch?, fov?, label?}] → numbered contact sheet."""
    def one(it):
        m = it["meta"]
        return render(m, it["heading"], it.get("pitch", 0), it.get("fov", 90), 640, 480,
                      proxy=proxy, cache=cache)
    with ThreadPoolExecutor(8) as ex:
        ims = list(ex.map(one, items))
    rows = (len(items) + cols - 1) // cols
    S = Image.new("RGB", (cols * tw, max(1, rows) * th), "black")
    d = ImageDraw.Draw(S)
    f = _font(16)
    for i, (it, im) in enumerate(zip(items, ims)):
        x, y = (i % cols) * tw, (i // cols) * th
        S.paste(im.resize((tw, th)), (x, y))
        d.rectangle([x, y, x + tw, y + 22], fill="black")
        d.text((x + 4, y + 2), it.get("label") or f"{i}: …{it['meta']['id'][-8:]} h{it['heading']:.0f}", fill="yellow", font=f)
    S.save(out, quality=88)


# --- scan for match.py ----------------------------------------------------------------------------------------------

def scan(lat: float, lon: float, radius_m: float, step_m: float, workers: int = 16, proxy: str | None = None) -> dict:
    """Query the nearest panorama point at each point of a square grid; deduplicate -> {id: {wgs, road, date}} (match.py rank --panos)."""
    pts = []
    n = int(radius_m // step_m)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            pts.append(geo.dest(geo.dest((lat, lon), 0, i * step_m), 90, j * step_m))
    found: dict = {}
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(lambda p: near(*p, radius=step_m, proxy=proxy), pts):
            if r:
                found[r["id"]] = r
    return {pid: {"wgs": p["wgs"], "road": p.get("street") or "", "date": p.get("date") or ""} for pid, p in found.items()}


def thin(panos: dict, spread_m: float) -> dict:
    """Thin by minimum spacing: of points closer than spread_m, keep only the first."""
    kept: dict = {}
    for k, v in panos.items():
        if all(geo.distance(tuple(v["wgs"]), tuple(u["wgs"])) >= spread_m for u in kept.values()):
            kept[k] = v
    return kept


# --- CLI -----------------------------------------------------------------------------------------------------------

def _neg_coords(argv: list[str]) -> list[str]:
    """argparse treats negative coordinates like -1.45,-48.5 as option names; prefixing a space makes them plain values (float ignores the space). Needed for every puzzle in the southern or western hemisphere."""
    return [" " + a if re.match(r"^-\d[\d.]*(,-?[\d.]+)+$", a) else a for a in argv]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy", default=os.environ.get("GEO_PROXY"), help=PROXY_HELP)
    ap.add_argument("--cache", type=Path, default=Path(".geo-cache/yandex"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--proxy", default=argparse.SUPPRESS, help="can go before or after the subcommand")
        sp.add_argument("--cache", type=Path, default=argparse.SUPPRESS)

    n = sub.add_parser("near")
    common(n)
    n.add_argument("latlon")
    n.add_argument("--radius", type=float, default=50)

    i = sub.add_parser("info")
    common(i)
    i.add_argument("id")

    r = sub.add_parser("render")
    common(r)
    r.add_argument("id")
    r.add_argument("--heading", type=float, required=True)
    r.add_argument("--pitch", type=float, default=0, help="positive = looking up")
    r.add_argument("--fov", type=float, default=90, help="horizontal field of view")
    r.add_argument("--width", type=int, default=1024)
    r.add_argument("--height", type=int, default=768)
    r.add_argument("--zoom", type=int, default=2, help="panorama resolution level (0 = largest); 2 is enough for comparison sheets")
    r.add_argument("--out", type=Path, required=True)

    sc = sub.add_parser("scan", help="every panorama point around a center -> .json for match.py rank --panos")
    common(sc)
    sc.add_argument("center")
    sc.add_argument("--radius", type=float, default=250)
    sc.add_argument("--step", type=float, default=25, help="grid step, m (also the nearest-point accept radius)")
    sc.add_argument("--spread", type=float, help="thinning: minimum meters between two points")
    sc.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("sheet")
    common(s)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", help="comma-separated panoids")
    g.add_argument("--at", help="lat,lon: take the nearest panorama point")
    g.add_argument("--points", type=Path, help="JSON {name:[lat,lon]}: take the nearest panorama point for each")
    h = s.add_mutually_exclusive_group(required=True)
    h.add_argument("--heading", type=float)
    h.add_argument("--headings", help="comma-separated, e.g. 0,60,120,180,240,300")
    h.add_argument("--toward", help="lat,lon: every point faces this target")
    s.add_argument("--offset", type=float, default=0)
    s.add_argument("--pitch", type=float, default=0)
    s.add_argument("--fov", type=float, default=90)
    s.add_argument("--radius", type=float, default=50)
    s.add_argument("--date", help="only this capture (2018 or 2018-07), picked from this panorama and its historical captures")
    s.add_argument("--limit", type=int, default=12)
    s.add_argument("--out", type=Path, required=True)

    args = ap.parse_args(_neg_coords(sys.argv[1:]))

    def _meta(pid: str) -> dict:
        try:
            m = info(pid, args.proxy, args.cache)
        except ServiceError as e:
            sys.exit(f"Yandex Panoramas lookup failed ({e}); run doctor.py --network to check service access")
        if not m:
            sys.exit(f"no Yandex panorama with id {pid}")
        return m

    if args.cmd == "near":
        lat, lon = map(float, args.latlon.split(","))
        try:
            res = near(lat, lon, args.radius, args.proxy)
        except ServiceError as e:
            sys.exit(f"Yandex Panoramas lookup failed ({e}); run doctor.py --network to check service access")
        print(json.dumps(res, ensure_ascii=False, indent=1) if res else f"no Yandex panorama within {args.radius:.0f} m (increase --radius or check coverage)")
    elif args.cmd == "info":
        print(json.dumps(_meta(args.id), ensure_ascii=False, indent=1))
    elif args.cmd == "render":
        m = _meta(args.id)
        render(m, args.heading, args.pitch, args.fov, args.width, args.height, args.zoom,
               args.proxy, args.cache).save(args.out)
        print(args.out)
    elif args.cmd == "scan":
        lat, lon = map(float, args.center.split(","))
        try:
            found = scan(lat, lon, args.radius, args.step, proxy=args.proxy)
        except ServiceError as e:
            sys.exit(f"Yandex Panoramas lookup failed ({e}); run doctor.py --network to check service access")
        if args.spread:
            found = thin(found, args.spread)
        args.out.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(found)} panorama points -> {args.out}")
    else:  # sheet
        metas: dict[str, dict] = {}
        if args.ids:
            for pid in args.ids.split(","):
                metas[pid] = _meta(pid)
        else:
            pts = {"at": list(map(float, args.at.split(",")))} if args.at else json.loads(args.points.read_text(encoding="utf-8"))
            for name, (la, lo) in pts.items():
                try:
                    res = near(la, lo, args.radius, args.proxy)
                except ServiceError as e:
                    sys.exit(f"Yandex Panoramas lookup failed ({e}); run doctor.py --network to check service access")
                if res and args.date:
                    p = pick_date(res, args.date)
                    if not p:
                        print(f"{name}: no {args.date} capture (available: {', '.join(res['dates_seen'])})", file=sys.stderr)
                        continue
                    metas.setdefault(p["id"], _meta(p["id"]))
                elif res:
                    metas.setdefault(res["id"], res)
                else:
                    print(f"{name}: no panorama nearby", file=sys.stderr)
        target = tuple(map(float, args.toward.split(","))) if args.toward else None
        items = []
        for pid, m in metas.items():
            if args.headings:
                heads = [float(x) for x in args.headings.split(",")]
            elif target:
                heads = [geo.bearing(tuple(m["wgs"]), target) + args.offset]
            else:
                heads = [args.heading]
            for hd in heads:
                label = " ".join(x for x in (f"{m.get('street', '')[:18]}", f"{m['wgs'][0]:.5f},{m['wgs'][1]:.5f}",
                                             m.get("date") or "", f"h{hd % 360:.0f}") if x)
                items.append({"meta": m, "heading": hd % 360, "pitch": args.pitch, "fov": args.fov,
                              "label": f"{len(items)}: {label}"})
        pages = [items[k:k + args.limit] for k in range(0, len(items), args.limit)] or [[]]
        for pi, page in enumerate(pages):
            out = args.out if pi == 0 else args.out.with_name(f"{args.out.stem}_{pi + 1}{args.out.suffix}")
            sheet(page, out, args.proxy, args.cache)
            print(out)
        index = [{k: v for k, v in it.items() if k != "meta"} | {"id": it["meta"]["id"]} for it in items]
        args.out.with_suffix(".index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    # Chinese-locale Windows writes GBK by default: ñ and m² make it crash, and the Russian the agent reads comes out garbled
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    main()
