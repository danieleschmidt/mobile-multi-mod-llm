"""Tests for core functionality."""

import numpy as np
import pytest

from mobile_multimodal import MobileMultiModalLLM


class TestMobileMultiModalLLM:
    """Test cases for MobileMultiModalLLM."""

    def test_from_pretrained(self):
        """Test model loading."""
        model = MobileMultiModalLLM.from_pretrained("test-model")
        assert model is not None
        assert model.model_path == "test-model"

    def test_generate_caption(self):
        """Test caption generation."""
        model = MobileMultiModalLLM()
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        caption = model.generate_caption(image)
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_extract_text(self):
        """Test text extraction."""
        model = MobileMultiModalLLM()
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        text_regions = model.extract_text(image)
        assert isinstance(text_regions, list)
        assert len(text_regions) > 0
        assert "text" in text_regions[0]
        assert "bbox" in text_regions[0]

    def test_answer_question(self):
        """Test visual question answering."""
        model = MobileMultiModalLLM()
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        answer = model.answer_question(image, "What is in the image?")
        assert isinstance(answer, str)
        assert len(answer) > 0