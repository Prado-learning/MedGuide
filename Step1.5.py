from flask import Flask, request, jsonify
import requests
import json
import re

app = Flask(__name__)
app.json.ensure_ascii = False

# 连接到大模型的API
url = "https://9zekn5682505.vicp.fun/v1/chat/completions"
headers = {'Content-Type': 'application/json'}
model_name = "BUPTmodel-70B"

def call_model(prompt):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 128
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=5)
        if resp.status_code != 200:
            return "无法生成引导问题"
        resp_data = resp.json()
        return resp_data["choices"][0]["message"]["content"].strip()
    except:
        return "无法生成引导问题"

@app.route('/api', methods=['POST'])
def api_endpoint():
    if request.method == 'POST':
        data = request.json
        question = data.get("question", "").strip()
        diagnosis = data.get("diagnosis", "").strip()

        if diagnosis == "不适用":
            prompt = (
                "你是一名康复医疗领域的专业医师。\n\n"
                "患者提供了如下描述，但当前无法确定诊断：\n"
                f"{question}\n\n"
                "请生成一个或多个有针对性的引导性问题，以帮助更好地判断患者的病情。\n"
                "请直接输出问题内容，不要添加额外的解释。"
            )
            guidance_question = call_model(prompt)
            return jsonify({"question": question, "guidance_question": guidance_question}), 200
        
        return jsonify({"question": question, "guidance_question": "不需要额外引导问题"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
