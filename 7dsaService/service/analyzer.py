import spacy
from transformers import pipeline
from typing import List

class SevenDimAnalyzer:
    def __init__(self, nlp_model="en_core_web_sm", device="cpu"):
        self.nlp = spacy.load(nlp_model)
        self.device = device
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased", device=self.device)

    def analyze_ethics(self, text: str) -> float:
        sentiment = self.sentiment_pipeline(text)[0]
        if sentiment['label'] == 'NEGATIVE':
            return -sentiment['score']
        else:
            return sentiment['score']

    def analyze_social_impact(self, text: str) -> float:
        keywords = ["community", "equality", "inclusion", "accessibility", "well-being", "education"]
        return self._keyword_analysis(text, keywords)

    def analyze_environmental_impact(self, text: str) -> float:
        keywords = ["sustainability", "conservation", "pollution", "climate change", "renewable energy", "waste reduction"]
        return self._keyword_analysis(text, keywords)

    def analyze_stakeholder_value(self, text: str) -> float:
        keywords = ["transparency", "accountability", "participation", "engagement", "feedback", "collaboration"]
        return self._keyword_analysis(text, keywords)

    def analyze_legal_compliance(self, text: str) -> float:
        keywords = ["compliance", "regulation", "privacy", "security", "liability", "governance"]
        return self._keyword_analysis(text, keywords)

    def analyze_text(self, text: str) -> dict:
        return {
            "ethics": self.analyze_ethics(text),
            "social_impact": self.analyze_social_impact(text),
            "environmental_impact": self.analyze_environmental_impact(text),
            "stakeholder_value": self.analyze_stakeholder_value(text),
            "legal_compliance": self.analyze_legal_compliance(text)
        }

    def _keyword_analysis(self, text: str, keywords: List[str]) -> float:
        doc = self.nlp(text)
        keyword_scores = [keyword in text.lower() for keyword in keywords]
        return sum(keyword_scores) / len(keywords)
