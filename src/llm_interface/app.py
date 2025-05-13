import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # For handling Cross-Origin Resource Sharing if frontend and backend are on different ports during development

API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

MODEL_ID = "google/txgemma-9b-chat"

# TODO: HF doesn't have inference API for txgemma at the moment
#       May need to switch to vLLM
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app) # Enable CORS for all routes


def query_huggingface_model(payload):
    """
    Sends a payload to the Hugging Face Inference API for the specified model.
    """
    if not API_TOKEN
        return {"error": "Hugging Face API token is not configured. Please set it in the backend."}

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120) # Increased timeout for large models
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to Hugging Face API: {e}"
        if response is not None: # Check if response object exists
             error_message += f" - Status Code: {response.status_code} - Response: {response.text}"
        return {"error": error_message}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


@app.route('/api/generate', methods=['POST'])
def generate_text():
    """
    API endpoint to receive a prompt and return the model's generation.
    """
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "No prompt provided or invalid JSON"}), 400

        prompt = data['prompt']

        payload = {
            "inputs": [
                {"role": "user", "content": prompt}
            ],
            "parameters": { # Optional
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            }
        }

        api_response = query_huggingface_model(payload)

        if "error" in api_response:
            error_detail = api_response.get("error", "Unknown error from Hugging Face API")
            if "estimated_time" in api_response:
                error_detail += f". Model might be loading (estimated time: {api_response['estimated_time']:.2f}s)."
            return jsonify({"error": error_detail}), 500

        generated_content = ""
        if isinstance(api_response, list) and api_response:
            last_message = api_response[0].get("generated_text")
            if isinstance(last_message, list) and last_message:
                assistant_reply = next((msg.get("content") for msg in reversed(last_message) if msg.get("role") == "assistant"), None)
                if assistant_reply:
                    generated_content = assistant_reply
                else:
                    generated_content = str(last_message) 
            elif isinstance(last_message, str):
                generated_content = last_message
            else: 
                 generated_content = str(api_response)
        elif isinstance(api_response, dict) and "generated_text" in api_response:
            generated_content = api_response["generated_text"]
        else:
            app.logger.warning(f"Unexpected API response format: {api_response}")
            return jsonify({"error": "Unexpected response format from model.", "details": api_response}), 500

        return jsonify({"generated_text": generated_content.strip()})

    except Exception as e:
        app.logger.error(f"Error in /api/generate: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Created 'static' folder. Please put your HTML file (e.g., index.html) there.")

    app.run(host='0.0.0.0', port=8888, debug=True)

