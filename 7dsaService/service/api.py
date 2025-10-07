from flask import Flask, request, jsonify
from service.analyzer import SevenDimAnalyzer

app = Flask(__name__)
analyzer = SevenDimAnalyzer()

@app.route("/analyze", methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data['text']
    results = analyzer.analyze_text(text)
    return jsonify(results)
