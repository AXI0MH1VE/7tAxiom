#!/usr/bin/env python3
"""
Comprehensive test suite for 7DSA Service.

Tests cover:
1. SevenDimAnalyzer functionality
2. Individual analysis dimensions
3. API endpoints
4. Error handling and edge cases
5. Integration tests
"""

import unittest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from service.analyzer import SevenDimAnalyzer
from service.api import app


class TestSevenDimAnalyzer(unittest.TestCase):
    """Test cases for SevenDimAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SevenDimAnalyzer()

        # Sample texts for testing
        self.positive_text = "This product promotes equality, sustainability, and community well-being through renewable energy and transparency."
        self.negative_text = "The company prioritizes profit over environmental concerns and stakeholder rights."
        self.neutral_text = "The business operates according to standard industry practices."

    def test_analyzer_initialization(self):
        """Test analyzer initialization with different configurations."""
        # Default initialization
        analyzer = SevenDimAnalyzer()
        self.assertIsNotNone(analyzer.nlp)
        self.assertIsNotNone(analyzer.sentiment_pipeline)
        self.assertEqual(analyzer.device, "cpu")

        # Custom initialization
        analyzer_custom = SevenDimAnalyzer(nlp_model="en_core_web_sm", device="cpu")
        self.assertEqual(analyzer_custom.device, "cpu")

    def test_analyze_ethics(self):
        """Test ethics analysis."""
        # Test positive text
        ethics_score = self.analyzer.analyze_ethics(self.positive_text)
        self.assertIsInstance(ethics_score, float)
        self.assertGreaterEqual(ethics_score, -1.0)
        self.assertLessEqual(ethics_score, 1.0)

        # Test negative text
        ethics_score_neg = self.analyzer.analyze_ethics(self.negative_text)
        self.assertIsInstance(ethics_score_neg, float)

        # Test neutral text
        ethics_score_neutral = self.analyzer.analyze_ethics(self.neutral_text)
        self.assertIsInstance(ethics_score_neutral, float)

    def test_analyze_social_impact(self):
        """Test social impact analysis."""
        social_score = self.analyzer.analyze_social_impact(self.positive_text)
        self.assertIsInstance(social_score, float)
        self.assertGreaterEqual(social_score, 0.0)
        self.assertLessEqual(social_score, 1.0)

        # Text with social keywords should score higher
        social_text = "community equality inclusion accessibility well-being education"
        social_score_keywords = self.analyzer.analyze_social_impact(social_text)
        self.assertGreater(social_score_keywords, social_score)

    def test_analyze_environmental_impact(self):
        """Test environmental impact analysis."""
        env_score = self.analyzer.analyze_environmental_impact(self.positive_text)
        self.assertIsInstance(env_score, float)
        self.assertGreaterEqual(env_score, 0.0)
        self.assertLessEqual(env_score, 1.0)

        # Text with environmental keywords should score higher
        env_text = "sustainability conservation pollution climate change renewable energy waste reduction"
        env_score_keywords = self.analyzer.analyze_environmental_impact(env_text)
        self.assertGreater(env_score_keywords, env_score)

    def test_analyze_stakeholder_value(self):
        """Test stakeholder value analysis."""
        stakeholder_score = self.analyzer.analyze_stakeholder_value(self.positive_text)
        self.assertIsInstance(stakeholder_score, float)
        self.assertGreaterEqual(stakeholder_score, 0.0)
        self.assertLessEqual(stakeholder_score, 1.0)

    def test_analyze_legal_compliance(self):
        """Test legal compliance analysis."""
        legal_score = self.analyzer.analyze_legal_compliance(self.positive_text)
        self.assertIsInstance(legal_score, float)
        self.assertGreaterEqual(legal_score, 0.0)
        self.assertLessEqual(legal_score, 1.0)

    def test_analyze_text_complete(self):
        """Test complete text analysis."""
        results = self.analyzer.analyze_text(self.positive_text)

        # Check that all dimensions are present
        expected_dimensions = [
            "ethics", "social_impact", "environmental_impact",
            "stakeholder_value", "legal_compliance"
        ]

        for dimension in expected_dimensions:
            self.assertIn(dimension, results)
            self.assertIsInstance(results[dimension], float)

        # Check score ranges
        for dimension, score in results.items():
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_keyword_analysis_method(self):
        """Test the keyword analysis helper method."""
        keywords = ["test", "keyword", "analysis"]
        text = "This is a test with keyword and analysis content"

        score = self.analyzer._keyword_analysis(text, keywords)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Should find all keywords
        self.assertEqual(score, 1.0)

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        empty_results = self.analyzer.analyze_text("")
        self.assertIsInstance(empty_results, dict)

        # Empty text should return valid results (likely 0 or neutral scores)
        for dimension, score in empty_results.items():
            self.assertIsInstance(score, float)

    def test_long_text_handling(self):
        """Test handling of long text."""
        long_text = "sustainability " * 1000  # Very long text
        results = self.analyzer.analyze_text(long_text)
        self.assertIsInstance(results, dict)

        # Should complete without errors
        for dimension, score in results.items():
            self.assertIsInstance(score, float)


class TestAPI(unittest.TestCase):
    """Test cases for API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.app = app.test_client()
        self.app.testing = True

    def test_analyze_endpoint_success(self):
        """Test successful analysis request."""
        test_data = {"text": self.positive_text}
        response = self.app.post('/analyze',
                                data=json.dumps(test_data),
                                content_type='application/json')

        self.assertEqual(response.status_code, 200)

        # Check response structure
        data = json.loads(response.data)
        expected_dimensions = [
            "ethics", "social_impact", "environmental_impact",
            "stakeholder_value", "legal_compliance"
        ]

        for dimension in expected_dimensions:
            self.assertIn(dimension, data)
            self.assertIsInstance(data[dimension], float)

    def test_analyze_endpoint_missing_text(self):
        """Test analysis request with missing text."""
        test_data = {}  # Missing text field
        response = self.app.post('/analyze',
                                data=json.dumps(test_data),
                                content_type='application/json')

        self.assertEqual(response.status_code, 200)  # Flask returns 200 even for missing data

    def test_analyze_endpoint_invalid_json(self):
        """Test analysis request with invalid JSON."""
        response = self.app.post('/analyze',
                                data="invalid json",
                                content_type='application/json')

        self.assertEqual(response.status_code, 400)

    def test_analyze_endpoint_no_json(self):
        """Test analysis request with no JSON content."""
        response = self.app.post('/analyze',
                                data="plain text",
                                content_type='text/plain')

        self.assertEqual(response.status_code, 200)  # Flask handles gracefully

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.app.post('/analyze',
                                data=json.dumps({"text": "test"}),
                                content_type='application/json')

        # Check for CORS headers (Flask-CORS adds these)
        self.assertIn('Access-Control-Allow-Origin', response.headers)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete service."""

    def test_end_to_end_analysis(self):
        """Test complete analysis workflow."""
        # Test with various types of content
        test_cases = [
            "This sustainable product promotes equality and environmental conservation.",
            "The company values transparency, accountability, and stakeholder engagement.",
            "Poor governance and lack of compliance with regulations.",
            "Excellent social impact through community programs and education initiatives."
        ]

        for text in test_cases:
            results = self.analyzer.analyze_text(text)

            # Verify all dimensions are analyzed
            self.assertEqual(len(results), 5)

            # Verify scores are reasonable
            for dimension, score in results.items():
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, -1.0)
                self.assertLessEqual(score, 1.0)

    def test_consistency_across_calls(self):
        """Test that analysis is consistent across multiple calls."""
        text = "sustainability equality transparency compliance"

        results1 = self.analyzer.analyze_text(text)
        results2 = self.analyzer.analyze_text(text)

        # Results should be identical for the same input
        for dimension in results1:
            self.assertAlmostEqual(results1[dimension], results2[dimension], places=6)

    def test_different_analyzers_consistency(self):
        """Test that different analyzer instances produce similar results."""
        text = "sustainability community transparency"

        analyzer1 = SevenDimAnalyzer()
        analyzer2 = SevenDimAnalyzer()

        results1 = analyzer1.analyze_text(text)
        results2 = analyzer2.analyze_text(text)

        # Results should be very similar (allowing for minor numerical differences)
        for dimension in results1:
            self.assertAlmostEqual(results1[dimension], results2[dimension], places=4)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        # Test with None input (should handle gracefully)
        try:
            results = self.analyzer.analyze_text(None)
            # If it doesn't crash, check that we get valid results
            self.assertIsInstance(results, dict)
        except Exception as e:
            self.fail(f"Analyzer should handle None input gracefully, but got: {e}")

    def test_special_characters_handling(self):
        """Test handling of text with special characters."""
        special_text = "Sustainability ♻️ with émojis and spëcial châractërs! @#$%^&*()"
        results = self.analyzer.analyze_text(special_text)

        # Should handle special characters without crashing
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 5)

    def test_very_long_text(self):
        """Test handling of extremely long text."""
        long_text = "sustainability " * 10000  # Very long text
        results = self.analyzer.analyze_text(long_text)

        # Should complete without memory issues or timeouts
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 5)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
