from __future__ import annotations

from math import degrees, hypot

import cv2
import numpy as np

from ...application.preview_cancellation import raise_if_preview_cancelled
from ...contour_extractor import estimate_effective_polygon_width_px
from ...domain import PolygonData, compute_polygon_metrics
from . import detector as _md


def estimate_inward_direction_by_gradient_profile(
    image: np.ndarray,
    edge_point: tuple[float, float],
    normal: tuple[float, float],
    radius: int,
    *,
    min_confidence: float = 0.15,
) -> tuple[tuple[float, float] | None, float, np.ndarray | None]:
    """Decide which side of a bright edge is inside the conductor using 1D intensity asymmetry.

    Samples intensities along ``normal`` from ``-radius`` to ``+radius`` (in pixel steps).
    The side with a sharper falloff is treated as background; the softer side is inward.
    """
    if image.size == 0:
        return None, 0.0, None
    h, w = int(image.shape[0]), int(image.shape[1])
    r = max(1, int(radius))
    nrm = np.array(normal, dtype=np.float64)
    nn = float(np.linalg.norm(nrm))
    if nn < 1e-9:
        return None, 0.0, None
    nrm = nrm / nn
    cx, cy = float(edge_point[0]), float(edge_point[1])

    profile: list[float] = []
    for s in range(-r, r + 1):
        x = cx + nrm[0] * s
        y = cy + nrm[1] * s
        xi = int(round(np.clip(x, 0, w - 1)))
        yi = int(round(np.clip(y, 0, h - 1)))
        profile.append(float(image[yi, xi]))
    prof = np.array(profile, dtype=np.float64)
    lo = max(0, r - 3)
    hi = min(len(prof), r + 4)
    peak_idx = lo + int(np.argmax(prof[lo:hi]))

    left = prof[:peak_idx]
    right = prof[peak_idx + 1 :]

    def _sharpness(side: np.ndarray) -> float:
        if side.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(side))))

    sharp_l = _sharpness(left)
    sharp_r = _sharpness(right)
    denom = max(sharp_l, sharp_r, 1e-6)
    confidence = float(abs(sharp_l - sharp_r) / denom)

    if confidence < float(min_confidence):
        return None, confidence, prof

    # Sharper decay = background. If left is sharper, inward points along +normal.
    if sharp_l > sharp_r:
        inward = (float(nrm[0]), float(nrm[1]))
    else:
        inward = (-float(nrm[0]), -float(nrm[1]))
    inl = hypot(inward[0], inward[1])
    if inl > 1e-9:
        inward = (inward[0] / inl, inward[1] / inl)
    return inward, confidence, prof


def _unit(dx: float, dy: float) -> tuple[float, float] | None:
    d = hypot(dx, dy)
    if d < 1e-6:
        return None
    return (dx / d, dy / d)


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    lx, ly = x2 - x1, y2 - y1
    le = hypot(lx, ly)
    if le < 1e-6:
        return hypot(px - x1, py - y1)
    return abs(lx * (y1 - py) - ly * (x1 - px)) / le


def _segment_projection_interval(
    x1: float, y1: float, x2: float, y2: float, ox: float, oy: float, ux: float, uy: float
) -> tuple[float, float]:
    t0 = (x1 - ox) * ux + (y1 - oy) * uy
    t1 = (x2 - ox) * ux + (y2 - oy) * uy
    return (min(t0, t1), max(t0, t1))


def _interp_on_segment(
    x1: float, y1: float, x2: float, y2: float, t_start: float, t_end: float, t: float
) -> tuple[float, float]:
    if abs(t_end - t_start) < 1e-6:
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    a = (t - t_start) / (t_end - t_start)
    return (x1 + a * (x2 - x1), y1 + a * (y2 - y1))


def _parallel_angle_diff_deg(u1: tuple[float, float], u2: tuple[float, float]) -> float:
    dotp = abs(u1[0] * u2[0] + u1[1] * u2[1])
    dotp = min(1.0, max(0.0, dotp))
    return degrees(np.arccos(dotp))


def _quad_from_pair(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
    ux: float,
    uy: float,
    ox: float,
    oy: float,
    t_lo: float,
    t_hi: float,
) -> list[tuple[float, float]] | None:
    ta0, ta1 = _segment_projection_interval(ax1, ay1, ax2, ay2, ox, oy, ux, uy)
    tb0, tb1 = _segment_projection_interval(bx1, by1, bx2, by2, ox, oy, ux, uy)
    pa_lo = _interp_on_segment(ax1, ay1, ax2, ay2, ta0, ta1, t_lo)
    pa_hi = _interp_on_segment(ax1, ay1, ax2, ay2, ta0, ta1, t_hi)
    pb_lo = _interp_on_segment(bx1, by1, bx2, by2, tb0, tb1, t_lo)
    pb_hi = _interp_on_segment(bx1, by1, bx2, by2, tb0, tb1, t_hi)
    quad = [pa_lo, pa_hi, pb_hi, pb_lo]
    if len(quad) < 4:
        return None
    return quad


def _finalize_metal_strip_polygon(
    quad: list[tuple[float, float]],
    shape_hw: tuple[int, int],
    config: _md.MetalRecoveryConfig,
    poly_id: int,
) -> tuple[PolygonData | None, _md.MetalPolygonRecord | None]:
    h, w = shape_hw
    cnt = np.array(quad, dtype=np.float32).reshape(-1, 1, 2)
    cnt_i = np.round(cnt).astype(np.int32)
    area = abs(float(cv2.contourArea(cnt_i)))
    if area < 1.0:
        return None, None
    perimeter = float(cv2.arcLength(cnt_i, True))
    temp_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(temp_mask, [cnt_i], 0, 255, thickness=-1)
    width_px, _wm = estimate_effective_polygon_width_px(temp_mask, cnt_i)
    rect = cv2.minAreaRect(cnt_i)
    rw, rh = float(rect[1][0]), float(rect[1][1])
    width_px = _md.effective_conductor_width_px(width_px, rw, rh)
    length_px = float(max(rw, rh))
    straight = _md._straightness_metric(rw, rh, perimeter)
    b_touch = _md._border_touch(cnt_i, width=w, height=h)

    points = _md._contour_to_polygon(
        cnt_i,
        epsilon=config.epsilon_simplify,
        approx_enabled=config.approximation_enabled,
        min_angle_deg=config.min_polygon_angle_deg,
    )
    raw_pts = [(float(cnt_i[i][0][0]), float(cnt_i[i][0][1])) for i in range(len(cnt_i))]
    topo_pts = points if len(points) >= 3 else raw_pts
    topo_pts, valid, topo_reason = _md._repair_contour_polygon_for_topology(cnt_i, topo_pts, config, (h, w))
    if valid and len(topo_pts) >= 3:
        points = topo_pts
    if len(points) < 3:
        dev, ang_ok = 0.0, True
    else:
        dev, ang_ok = _md._angle_deviation_for_mode(
            points,
            allowed=config.allowed_angles,
            tolerance_deg=config.angle_tolerance_deg,
        )

    n_vertices = len(topo_pts)
    if n_vertices >= 2 and topo_pts[0] == topo_pts[-1]:
        n_vertices = max(0, n_vertices - 1)

    reject_case = ""
    t_branch_score = 0
    if not valid:
        reject_case = topo_reason or "некорректный_контур"
    elif width_px + 1e-3 < config.min_width_px:
        reject_case = "ширина"
    elif config.max_width_px is not None and width_px > float(config.max_width_px) + 1e-3:
        reject_case = "ширина_макс"
    elif length_px + 1e-3 < config.min_length_px:
        reject_case = "длина"
    elif area + 1e-3 < config.min_area:
        reject_case = "площадь"
    elif config.max_area is not None and area > float(config.max_area) + 1e-3:
        reject_case = "площадь_макс"
    elif perimeter + 1e-3 < config.min_perimeter:
        reject_case = "периметр"
    elif config.max_perimeter is not None and perimeter > float(config.max_perimeter) + 1e-3:
        reject_case = "периметр_макс"
    elif straight + 1e-4 < config.min_straightness:
        reject_case = "прямолинейность"
    elif not ang_ok:
        reject_case = "углы"
    elif not config.allow_t_junction:
        t_branch_score = _md._convex_branching_score(cnt_i)
        if t_branch_score > 2:
            reject_case = "т_соединение_запрещено"
    if not reject_case and config.border_mode == "ignore" and b_touch:
        reject_case = "граница"

    reject_reason = ""
    if reject_case:
        reject_reason = _md.format_metal_reject_detail_ru(
            reject_case,
            config=config,
            width_px=float(width_px),
            length_px=length_px,
            area=area,
            perimeter=perimeter,
            straightness=straight,
            max_angle_deviation=float(dev),
            vertex_count=n_vertices,
            t_branch_score=int(t_branch_score),
        )

    use_pts = points if len(points) >= 3 else raw_pts
    poly = PolygonData(
        id=poly_id,
        points=use_pts,
        is_hole=False,
        parent_id=None,
        category="metal_wide_gradient",
        shape_hint="polygon",
    )
    poly.area, poly.perimeter, poly.bbox = compute_polygon_metrics(poly.points)

    rec = _md.MetalPolygonRecord(
        polygon=poly,
        area=area,
        perimeter=perimeter,
        width_px=float(width_px),
        length_px=length_px,
        straightness=straight,
        max_angle_deviation=float(dev),
        contour_validity=valid,
        border_touch=b_touch,
        reject_reason=reject_reason,
    )
    if reject_reason:
        return None, rec
    return poly, rec


def _pair_outline_polygon(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
    pid: int,
) -> PolygonData:
    pts = [
        (ax1, ay1),
        (ax2, ay2),
        (bx2, by2),
        (bx1, by1),
    ]
    poly = PolygonData(
        id=pid,
        points=pts,
        is_hole=False,
        parent_id=None,
        category="wide_pair_debug",
        shape_hint="polygon",
    )
    poly.area, poly.perimeter, poly.bbox = compute_polygon_metrics(poly.points)
    return poly


def recover_wide_conductors_by_gradient(
    gray: np.ndarray,
    config: _md.MetalRecoveryConfig,
) -> tuple[list[PolygonData], dict[str, np.ndarray], dict[str, list[PolygonData]]]:
    """Detect wide low-contrast conductors from parallel bright edges (SEM-friendly)."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = int(g.shape[0]), int(g.shape[1])
    overlays: dict[str, list[PolygonData]] = {"wide_pairs_suspicious": [], "wide_pairs_rejected": []}
    dbg: dict[str, np.ndarray] = {}

    if g.size == 0:
        return [], dbg, overlays

    blur = cv2.GaussianBlur(g, (3, 3), 0)
    raise_if_preview_cancelled()
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    p92 = float(np.percentile(mag, 92.0)) if mag.size else 0.0
    thr = max(12.0, p92 * 0.85)
    edge = (mag >= thr).astype(np.uint8) * 255
    gap = max(0, int(config.wide_gradient_max_edge_gap_px))
    if gap > 0:
        k = max(3, gap * 2 + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, ker)
    dbg["metal_wide_edge_map"] = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    min_pair = float(max(config.wide_gradient_min_pair_length_px, config.min_length_px * 0.5))
    lines = cv2.HoughLinesP(
        edge,
        1,
        np.pi / 180.0,
        threshold=max(18, int(min_pair // 3)),
        minLineLength=max(12, int(min_pair * 0.5)),
        maxLineGap=max(1, gap),
    )
    raise_if_preview_cancelled()
    if lines is None:
        conf_vis = np.zeros((h, w), dtype=np.uint8)
        dbg["metal_wide_gradient_confidence"] = cv2.applyColorMap(conf_vis, cv2.COLORMAP_VIRIDIS)
        dbg["metal_wide_inward_dirs"] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        dbg["metal_wide_pair_candidates"] = dbg["metal_wide_inward_dirs"].copy()
        dbg["metal_wide_recovered"] = dbg["metal_wide_inward_dirs"].copy()
        dbg["metal_wide_final_overlay"] = dbg["metal_wide_inward_dirs"].copy()
        return [], dbg, overlays

    lines_arr = lines[:, 0, :].astype(np.float64)
    lengths = np.hypot(lines_arr[:, 2] - lines_arr[:, 0], lines_arr[:, 3] - lines_arr[:, 1])
    order = np.argsort(-lengths)
    max_lines = 140
    order = order[:max_lines]

    par_tol = float(config.wide_gradient_parallel_tolerance_deg)
    min_w = float(config.min_width_px)
    max_w = float(config.max_width_px) if config.max_width_px is not None else min_w * 40.0
    ov_min = float(config.wide_gradient_min_overlap_ratio)
    r_prof = max(1, int(config.wide_gradient_profile_radius_px))
    min_dir_c = float(config.wide_gradient_min_direction_confidence)

    inward_vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    conf_acc = np.zeros((h, w), dtype=np.float32)
    pair_vis = inward_vis.copy()
    recovered_vis = inward_vis.copy()

    accepted: list[PolygonData] = []
    next_dbg_id = -1000
    next_poly_id = 1

    def _draw_arrow(vis: np.ndarray, x: float, y: float, vx: float, vy: float, col: tuple[int, int, int]) -> None:
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + vx * (r_prof + 2))), int(round(y + vy * (r_prof + 2))))
        cv2.arrowedLine(vis, p1, p2, col, 1, tipLength=0.35)

    for ii in range(len(order)):
        if ii & 1 == 0:
            raise_if_preview_cancelled()
        i = int(order[ii])
        ax1, ay1, ax2, ay2 = lines_arr[i]
        ui = _unit(ax2 - ax1, ay2 - ay1)
        if ui is None:
            continue
        mi = ((ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5)
        for jj in range(ii + 1, len(order)):
            if jj & 15 == 0:
                raise_if_preview_cancelled()
            j = int(order[jj])
            bx1, by1, bx2, by2 = lines_arr[j]
            uj = _unit(bx2 - bx1, by2 - by1)
            if uj is None:
                continue
            if _parallel_angle_diff_deg(ui, uj) > par_tol + 0.5:
                continue

            ux, uy = ui[0], ui[1]
            mj = ((bx1 + bx2) * 0.5, (by1 + by2) * 0.5)
            vmx, vmy = mj[0] - mi[0], mj[1] - mi[1]
            nvec = (-uy, ux)
            if nvec[0] * vmx + nvec[1] * vmy < 0:
                nvec = (-nvec[0], -nvec[1])
            nn = hypot(nvec[0], nvec[1])
            if nn < 1e-9:
                continue
            nvec = (nvec[0] / nn, nvec[1] / nn)

            dist = _point_line_distance(mi[0], mi[1], bx1, by1, bx2, by2)
            if dist < min_w - 1.5 or dist > max_w + 1.5:
                continue

            ox, oy = mi[0], mi[1]
            ia = _segment_projection_interval(ax1, ay1, ax2, ay2, ox, oy, ux, uy)
            ib = _segment_projection_interval(bx1, by1, bx2, by2, ox, oy, ux, uy)
            t_lo = max(ia[0], ib[0])
            t_hi = min(ia[1], ib[1])
            overlap = t_hi - t_lo
            la = ia[1] - ia[0]
            lb = ib[1] - ib[0]
            if overlap <= 0 or la <= 1.0 or lb <= 1.0:
                continue
            if overlap < ov_min * min(la, lb):
                continue

            inward_i, ci, _ = estimate_inward_direction_by_gradient_profile(
                blur, mi, nvec, r_prof, min_confidence=0.0
            )
            inward_j, cj, _ = estimate_inward_direction_by_gradient_profile(
                blur, mj, (-nvec[0], -nvec[1]), r_prof, min_confidence=0.0
            )

            xi, yi = int(np.clip(round(mi[0]), 0, w - 1)), int(np.clip(round(mi[1]), 0, h - 1))
            xj, yj = int(np.clip(round(mj[0]), 0, w - 1)), int(np.clip(round(mj[1]), 0, h - 1))
            conf_acc[yi, xi] = max(conf_acc[yi, xi], ci)
            conf_acc[yj, xj] = max(conf_acc[yj, xj], cj)

            if inward_i is not None:
                _draw_arrow(inward_vis, mi[0], mi[1], inward_i[0], inward_i[1], (0, 255, 180))
            if inward_j is not None:
                _draw_arrow(inward_vis, mj[0], mj[1], inward_j[0], inward_j[1], (0, 255, 180))

            if ci < min_dir_c or cj < min_dir_c:
                po = _pair_outline_polygon(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, next_dbg_id)
                next_dbg_id -= 1
                overlays["wide_pairs_suspicious"].append(po)
                cv2.line(pair_vis, (int(ax1), int(ay1)), (int(ax2), int(ay2)), (0, 200, 255), 1)
                cv2.line(pair_vis, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 200, 255), 1)
                continue
            if inward_i is None or inward_j is None:
                po = _pair_outline_polygon(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, next_dbg_id)
                next_dbg_id -= 1
                overlays["wide_pairs_rejected"].append(po)
                cv2.line(pair_vis, (int(ax1), int(ay1)), (int(ax2), int(ay2)), (0, 0, 255), 1)
                cv2.line(pair_vis, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 1)
                continue
            facing = inward_i[0] * inward_j[0] + inward_i[1] * inward_j[1]
            if facing > -0.15:
                po = _pair_outline_polygon(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, next_dbg_id)
                next_dbg_id -= 1
                overlays["wide_pairs_rejected"].append(po)
                cv2.line(pair_vis, (int(ax1), int(ay1)), (int(ax2), int(ay2)), (0, 0, 255), 1)
                cv2.line(pair_vis, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 1)
                continue

            quad = _quad_from_pair(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, ux, uy, ox, oy, t_lo, t_hi)
            if quad is None:
                continue
            poly, rec = _finalize_metal_strip_polygon(quad, (h, w), config, next_poly_id)
            if poly is None:
                po = _pair_outline_polygon(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2, next_dbg_id)
                next_dbg_id -= 1
                overlays["wide_pairs_rejected"].append(po)
                continue
            next_poly_id += 1
            accepted.append(poly)
            cnt_i = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(recovered_vis, [cnt_i], 0, (255, 120, 0), 2)
            cv2.line(pair_vis, (int(ax1), int(ay1)), (int(ax2), int(ay2)), (0, 255, 0), 1)
            cv2.line(pair_vis, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 1)

    cnorm = conf_acc.copy()
    if float(np.max(cnorm)) > 1e-6:
        cnorm = (cnorm / float(np.max(cnorm)) * 255.0).astype(np.uint8)
    else:
        cnorm = np.zeros((h, w), dtype=np.uint8)
    dbg["metal_wide_gradient_confidence"] = cv2.applyColorMap(cnorm, cv2.COLORMAP_VIRIDIS)
    dbg["metal_wide_inward_dirs"] = inward_vis
    dbg["metal_wide_pair_candidates"] = pair_vis

    for p in accepted:
        cnt = np.array(p.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(recovered_vis, [cnt], 0, (255, 200, 60), -1)
    dbg["metal_wide_recovered"] = recovered_vis

    final_vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for p in accepted:
        cnt = np.array(p.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(final_vis, [cnt], 0, (255, 180, 0), 2)
    dbg["metal_wide_final_overlay"] = final_vis

    return accepted, dbg, overlays
