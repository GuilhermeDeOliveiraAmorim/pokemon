"""Image utilities: card bounds detection, region cropping, preprocessing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CardBounds:
    rect: tuple[int, int, int, int]  # x, y, w, h
    detected: bool = False
    tolerance: float = 0.0


@dataclass
class Regions:
    name: np.ndarray | None = None
    name_alt: np.ndarray | None = None
    bottom: np.ndarray | None = None
    footer: np.ndarray | None = None
    # Raw (unscaled) crops kept for on-demand variant generation — much smaller
    # than pre-computing all 5 variants upfront.
    footer_raw: np.ndarray | None = None
    number: np.ndarray | None = None
    number_raw: np.ndarray | None = None
    code: np.ndarray | None = None
    code_raw: np.ndarray | None = None
    source_width: int = 0
    source_height: int = 0
    width: int = 0
    height: int = 0
    card_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    card_found: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"nao foi possivel abrir imagem: {path}")
    return img


def preprocess_image(img: np.ndarray, contrast: float = 35.0) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        gray = np.clip(f * (gray.astype(np.float32) - 127) + 127, 0, 255).astype(np.uint8)
    return gray


def detect_card_bounds(img: np.ndarray) -> CardBounds:
    h, w = img.shape[:2]
    if w < 50 or h < 50:
        return CardBounds(rect=(0, 0, w, h))

    background = _estimate_background_color(img)
    tolerance = _estimate_background_tolerance(img, background)
    if tolerance < 28:
        tolerance = 28

    bg_candidate = _detect_by_background_contrast(img, background, tolerance)
    edge_candidate = _detect_by_edge_transitions(img)
    best = _choose_best_candidate(img, bg_candidate, edge_candidate)
    if best is None:
        return CardBounds(rect=(0, 0, w, h), tolerance=tolerance)
    return CardBounds(rect=best, detected=True, tolerance=tolerance)


def prepare_regions(image_path: str) -> Regions:
    img = load_image(image_path)
    sh, sw = img.shape[:2]

    bounds = detect_card_bounds(img)
    working = img
    if bounds.detected:
        x, y, rw, rh = bounds.rect
        working = img[y : y + rh, x : x + rw]

    h, w = working.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("imagem invalida")

    name_rect = (w * 8 // 100, h * 4 // 100, w * 72 // 100, h * 16 // 100)
    name_alt_rect = (w * 80 // 1684, h * 10 // 2359, w * 1130 // 1684, h * 230 // 2359)
    bottom_rect = (w * 5 // 100, h * 86 // 100, w * 95 // 100, h * 98 // 100)
    footer_rect = (w * 8 // 100, h * 91 // 100, w * 36 // 100, h * 98 // 100)
    number_rect = (0, h * 1940 // 2359, w * 900 // 1684, h)
    code_rect = (0, h * 87 // 100, w * 42 // 100, h * 97 // 100)

    name_img = _crop(working, name_rect)
    name_alt_img = _scale_nearest(_crop(working, name_alt_rect), 4)
    bottom_img = _crop(working, bottom_rect)

    footer_raw = _crop(working, footer_rect)
    footer_img = _binarize(_scale_nearest(footer_raw, 6), 150)

    number_raw = _crop(working, number_rect)
    number_img = number_raw

    code_raw = _crop(working, code_rect)
    code_img = _binarize(_scale_nearest(code_raw, 6), 145)

    return Regions(
        name=name_img,
        name_alt=name_alt_img,
        bottom=bottom_img,
        footer=footer_img,
        footer_raw=footer_raw,
        number=number_img,
        number_raw=number_raw,
        code=code_img,
        code_raw=code_raw,
        source_width=sw,
        source_height=sh,
        width=w,
        height=h,
        card_rect=bounds.rect,
        card_found=bounds.detected,
    )


def preprocess_and_save(input_path: str, output_path: str, contrast: float = 35.0) -> dict:
    img = load_image(input_path)
    sh, sw = img.shape[:2]
    bounds = detect_card_bounds(img)
    working = img
    if bounds.detected:
        x, y, rw, rh = bounds.rect
        working = img[y : y + rh, x : x + rw]
    processed = preprocess_image(working, contrast)
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, processed)
    ph, pw = processed.shape[:2]
    return {
        "source_path": input_path,
        "output_path": output_path,
        "source_width": sw,
        "source_height": sh,
        "processed_width": pw,
        "processed_height": ph,
        "card_rect": bounds.rect,
        "card_found": bounds.detected,
        "contrast": contrast,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _crop(img: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return img[y1:y2, x1:x2].copy()


def _scale_nearest(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def _binarize(img: np.ndarray, threshold: int) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def _binarize_adaptive(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


def _enhance_clahe(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _make_ocr_variants(img: np.ndarray, scale: int = 4):
    """Yield preprocessed variants of a region one at a time for OCR attempts.

    Using a generator avoids allocating all 5 arrays simultaneously — the
    caller (early-exit loop) will stop consuming as soon as a match is found,
    so later variants are never even computed.
    """
    scaled = _scale_nearest(img, scale)
    clahe = _enhance_clahe(scaled)
    # 1. CLAHE + adaptive threshold
    yield _binarize_adaptive(clahe)
    # 2. CLAHE + Otsu threshold
    yield _otsu_binarize(clahe)
    # 3. Inverted
    yield cv2.bitwise_not(_binarize_adaptive(clahe))
    # 4. High contrast fixed threshold
    yield _binarize(scaled, 120)
    # 5. Lower threshold for light backgrounds
    yield _binarize(scaled, 170)


def _otsu_binarize(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _sampling_step(w: int, h: int) -> int:
    longer = max(w, h)
    step = longer // 300
    return max(2, min(step, 8))


def _estimate_background_color(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    dx, dy = w // 20, h // 20
    points = [
        (dx, dy),
        (w - 1 - dx, dy),
        (dx, h - 1 - dy),
        (w - 1 - dx, h - 1 - dy),
        (w // 2, dy),
        (w // 2, h - 1 - dy),
        (dx, h // 2),
        (w - 1 - dx, h // 2),
    ]
    colors = [img[y, x].astype(np.float64) for x, y in points]
    return np.mean(colors, axis=0)


def _color_distance(pixel: np.ndarray, background: np.ndarray) -> float:
    diff = pixel.astype(np.float64) - background
    return float(np.sqrt(np.sum(diff * diff)))


def _estimate_background_tolerance(img: np.ndarray, background: np.ndarray) -> float:
    h, w = img.shape[:2]
    step = _sampling_step(w, h)
    total = 0.0
    count = 0
    for x in range(0, w, step):
        total += _color_distance(img[0, x], background)
        total += _color_distance(img[h - 1, x], background)
        count += 2
    for y in range(step, h - step, step):
        total += _color_distance(img[y, 0], background)
        total += _color_distance(img[y, w - 1], background)
        count += 2
    if count == 0:
        return 32.0
    return (total / count) + 18


def _detect_by_background_contrast(
    img: np.ndarray, background: np.ndarray, tolerance: float
) -> tuple[int, int, int, int] | None:
    h, w = img.shape[:2]
    step = _sampling_step(w, h)
    min_x, min_y = w, h
    max_x, max_y = -1, -1
    fg_samples = 0

    for y in range(0, h, step):
        for x in range(0, w, step):
            if _color_distance(img[y, x], background) > tolerance:
                fg_samples += 1
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if fg_samples == 0:
        return None

    pad = max(step * 2, 12)
    x1 = max(0, min_x - pad)
    y1 = max(0, min_y - pad)
    x2 = min(w, max_x + step + pad)
    y2 = min(h, max_y + step + pad)

    rw, rh = x2 - x1, y2 - y1
    area_ratio = (rw * rh) / (w * h)
    if area_ratio < 0.18 or area_ratio > 0.98:
        return None
    density = (fg_samples * step * step) / (rw * rh)
    if density < 0.35:
        return None
    return (x1, y1, rw, rh)


def _luminance_at(img: np.ndarray, x: int, y: int) -> float:
    h, w = img.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    px = img[y, x]
    if len(img.shape) == 3:
        return float(0.299 * px[2] + 0.587 * px[1] + 0.114 * px[0])
    return float(px)


def _edge_strength_h(img: np.ndarray, x: int, y: int, window: int) -> float:
    total = 0.0
    for off in (-1, 0, 1):
        total += abs(_luminance_at(img, x - window // 2, y + off) - _luminance_at(img, x + window // 2, y + off))
    return total / 3.0


def _edge_strength_v(img: np.ndarray, x: int, y: int, window: int) -> float:
    total = 0.0
    for off in (-1, 0, 1):
        total += abs(_luminance_at(img, x + off, y - window // 2) - _luminance_at(img, x + off, y + window // 2))
    return total / 3.0


def _estimate_edge_noise(img: np.ndarray, window: int) -> float:
    h, w = img.shape[:2]
    samples = []
    cx, cy = w // 2, h // 2
    for x in range(cx - window * 2, cx + window * 2, window):
        for y in range(cy - window * 2, cy + window * 2, window):
            if 0 <= x < w and 0 <= y < h:
                samples.append(_edge_strength_h(img, x, y, window))
    return float(np.median(samples)) if samples else 5.0


def _detect_by_edge_transitions(img: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = img.shape[:2]
    window = max(_sampling_step(w, h) * 2, 4)
    row_step = max(h // 70, window)
    col_step = max(w // 70, window)
    edge_threshold = max(14.0, _estimate_edge_noise(img, window) * 2.8)

    left_edges, right_edges = [], []
    for y in range(h // 12, h - h // 12, row_step):
        e = _find_strong_h_edge(img, y, window, edge_threshold, from_left=True)
        if e is not None:
            left_edges.append(e)
        e = _find_strong_h_edge(img, y, window, edge_threshold, from_left=False)
        if e is not None:
            right_edges.append(e)

    top_edges, bottom_edges = [], []
    for x in range(w // 12, w - w // 12, col_step):
        e = _find_strong_v_edge(img, x, window, edge_threshold, from_top=True)
        if e is not None:
            top_edges.append(e)
        e = _find_strong_v_edge(img, x, window, edge_threshold, from_top=False)
        if e is not None:
            bottom_edges.append(e)

    min_support = 4
    if any(len(lst) < min_support for lst in (left_edges, right_edges, top_edges, bottom_edges)):
        return None

    def median(lst):
        s = sorted(lst)
        n = len(s)
        return s[n // 2]

    pad = max(window, 10)
    x1 = max(0, median(left_edges) - window - pad)
    y1 = max(0, median(top_edges) - window - pad)
    x2 = min(w, median(right_edges) + window + pad)
    y2 = min(h, median(bottom_edges) + window + pad)

    rw, rh = x2 - x1, y2 - y1
    area_ratio = (rw * rh) / (w * h)
    if area_ratio < 0.18 or area_ratio > 0.98:
        return None
    aspect = rw / rh if rh > 0 else 0
    if aspect < 0.48 or aspect > 0.82:
        return None
    return (x1, y1, rw, rh)


def _find_strong_h_edge(
    img: np.ndarray, y: int, window: int, threshold: float, from_left: bool
) -> int | None:
    h_img, w_img = img.shape[:2]
    search_width = w_img * 35 // 100
    best_strength = 0.0
    best_x = 0
    if from_left:
        for x in range(window, search_width, window):
            if x + window >= w_img:
                break
            s = _edge_strength_h(img, x, y, window)
            if s > best_strength:
                best_strength = s
                best_x = x
            if s >= threshold:
                return x
    else:
        for x in range(w_img - 1 - window, w_img - search_width, -window):
            if x - window < 0:
                break
            s = _edge_strength_h(img, x, y, window)
            if s > best_strength:
                best_strength = s
                best_x = x
            if s >= threshold:
                return x
    if best_strength >= threshold * 1.2:
        return best_x
    return None


def _find_strong_v_edge(
    img: np.ndarray, x: int, window: int, threshold: float, from_top: bool
) -> int | None:
    h_img, w_img = img.shape[:2]
    search_height = h_img * 35 // 100
    best_strength = 0.0
    best_y = 0
    if from_top:
        for y in range(window, search_height, window):
            if y + window >= h_img:
                break
            s = _edge_strength_v(img, x, y, window)
            if s > best_strength:
                best_strength = s
                best_y = y
            if s >= threshold:
                return y
    else:
        for y in range(h_img - 1 - window, h_img - search_height, -window):
            if y - window < 0:
                break
            s = _edge_strength_v(img, x, y, window)
            if s > best_strength:
                best_strength = s
                best_y = y
            if s >= threshold:
                return y
    if best_strength >= threshold * 1.2:
        return best_y
    return None


def _choose_best_candidate(
    img: np.ndarray,
    bg_candidate: tuple[int, int, int, int] | None,
    edge_candidate: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if bg_candidate is None and edge_candidate is None:
        return None
    if bg_candidate is None:
        return edge_candidate
    if edge_candidate is None:
        return bg_candidate

    h, w = img.shape[:2]
    total = w * h

    def score(rect):
        _, _, rw, rh = rect
        return (rw * rh) / total

    if score(bg_candidate) >= score(edge_candidate):
        return bg_candidate
    return edge_candidate
