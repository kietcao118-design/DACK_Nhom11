import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv() 
app = Flask(__name__)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or ""
client = InferenceClient(api_key=HF_TOKEN.strip())
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"reply": "Lỗi: Không nhận được dữ liệu JSON."})
     
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Vui lòng nhập câu hỏi!"})
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý ảo hữu ích của Nhóm 11. Trả lời bằng tiếng Việt."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500
        ) 
        reply_text = response.choices[0].message.content
        return jsonify({"reply": str(reply_text)})
    except Exception as e:
        print(f"--- LỖI HỆ THỐNG ---: {str(e)}")
        error_msg = str(e)
        if "ascii" in error_msg.lower():
            return jsonify({"reply": "Loi ma hoa (Encoding Error). Vui long chay lenh 'export PYTHONIOENCODING=utf-8' trong Terminal rồi chạy lại app."})
        
        return jsonify({"reply": f"Lỗi gọi Model: {error_msg}"})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)