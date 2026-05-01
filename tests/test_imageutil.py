"""Tests for imageutil module."""

import numpy as np
import pytest

from src.imageutil import (
    detect_card_bounds,
    preprocess_image,
    _crop,
    _scale_nearest,
    _binarize,
)


class TestPreprocessImage:
    def test_grayscale_output(self):
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = preprocess_image(img, contrast=0)
        assert len(result.shape) == 2
        assert result.shape == (100, 80)

    def test_contrast_applied(self):
        img = np.full((100, 80, 3), 128, dtype=np.uint8)
        result = preprocess_image(img, contrast=35)
        assert result.shape == (100, 80)


class TestCrop:
    def test_basic_crop(self):
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        cropped = _crop(img, (10, 10, 50, 50))
        assert cropped.shape == (40, 40, 3)

    def test_clamp_to_bounds(self):
        img = np.zeros((100, 80, 3), dtype=np.uint8)
        cropped = _crop(img, (0, 0, 200, 200))
        assert cropped.shape == (100, 80, 3)


class TestScaleNearest:
    def test_scale_2x(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        scaled = _scale_nearest(img, 2)
        assert scaled.shape == (20, 20, 3)


class TestBinarize:
    def test_binary_output(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = _binarize(img, 128)
        assert len(result.shape) == 2
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})


class TestDetectCardBounds:
    def test_small_image_not_detected(self):
        img = np.zeros((30, 30, 3), dtype=np.uint8)
        bounds = detect_card_bounds(img)
        assert not bounds.detected

    def test_card_on_background(self):
        # White background with dark card-like rectangle
        img = np.full((300, 200, 3), 240, dtype=np.uint8)
        img[50:250, 30:170, :] = 40  # dark card region
        bounds = detect_card_bounds(img)
        assert bounds.detected
        x, y, w, h = bounds.rect
        # The detected rect should roughly contain the card
        assert x <= 40
        assert y <= 60
        assert x + w >= 160
        assert y + h >= 240
