from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# 1. Dán Token MỚI đã cấp quyền Inference vào đây
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or "hf_Dán_Mã_Token_MỚI_Vừa_Lưu_Vào_Đây"

# 2. Khởi tạo Client (BỎ base_url để thư viện tự điều hướng thông minh nhất)
client = InferenceClient(api_key=HF_TOKEN)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    
    try:
        # Thay model thành Qwen2.5-72B-Instruct - Model Chat chuẩn nhất hiện nay
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct", # <--- THAY DÒNG NÀY
            messages=[
                {"role": "system", "content": "Bạn là trợ lý ảo hữu ích của Nhóm 11."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500
        )
        
        reply_text = response.choices[0].message.content
        return jsonify({"reply": reply_text})

    except Exception as e:
        # Nếu model 72B quá tải, hãy thử bản nhẹ hơn: "Qwen/Qwen2.5-7B-Instruct"
        return jsonify({"reply": f"Lỗi gọi Model: {str(e)}"})
    except Exception as e:
        # Nếu vẫn lỗi 403, hãy thử đổi model sang "microsoft/Phi-3-mini-4k-instruct"
        return jsonify({"reply": f"Lỗi quyền hạn (403): {str(e)}. Hãy kiểm tra lại mục 'Inference Providers' trong Token!"})

if __name__ == "__main__":
    app.run(debug=True)