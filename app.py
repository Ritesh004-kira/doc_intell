"""
Flask backend for Azure AI Document Intelligence UI
Wraps the prebuilt-read model with a web interface.
"""

import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import numpy as np

# Use absolute paths for templates and static folders for Vercel compatibility
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)
CORS(app)


def format_bounding_box(bounding_box):
    if not bounding_box:
        return "N/A"
    reshaped = np.array(bounding_box).reshape(-1, 2)
    return ", ".join(["[{:.2f}, {:.2f}]".format(x, y) for x, y in reshaped])


def build_result(result):
    """Convert Azure Document Intelligence result to a JSON-serializable dict."""
    output = {
        "content": result.content or "",
        "styles": [],
        "pages": [],
    }

    # Styles (handwritten check)
    if result.styles:
        for style in result.styles:
            output["styles"].append({
                "is_handwritten": style.is_handwritten or False,
            })

    # Pages
    if result.pages:
        for page in result.pages:
            page_data = {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "lines": [],
                "words": [],
            }

            if page.lines:
                for idx, line in enumerate(page.lines):
                    page_data["lines"].append({
                        "index": idx,
                        "content": line.content,
                        "bounding_box": format_bounding_box(line.polygon),
                    })

            if page.words:
                for word in page.words:
                    page_data["words"].append({
                        "content": word.content,
                        "confidence": round(word.confidence, 4) if word.confidence is not None else None,
                    })

            output["pages"].append(page_data)

    return output


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/analyze/url", methods=["POST"])
def analyze_url():
    data = request.get_json()
    endpoint = data.get("endpoint", "").strip()
    key = data.get("key", "").strip()
    url = data.get("url", "").strip()

    if not endpoint or not key or not url:
        return jsonify({"error": "endpoint, key, and url are required."}), 400

    try:
        client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        poller = client.begin_analyze_document(
            "prebuilt-read", AnalyzeDocumentRequest(url_source=url)
        )
        result = poller.result()
        return jsonify({"success": True, "result": build_result(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/file", methods=["POST"])
def analyze_file():
    endpoint = request.form.get("endpoint", "").strip()
    key = request.form.get("key", "").strip()
    file = request.files.get("file")

    if not endpoint or not key or not file:
        return jsonify({"error": "endpoint, key, and file are required."}), 400

    try:
        file_bytes = file.read()
        client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        poller = client.begin_analyze_document(
            "prebuilt-read",
            file_bytes,
            content_type=file.mimetype or "application/octet-stream",
        )
        result = poller.result()
        return jsonify({"success": True, "result": build_result(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
