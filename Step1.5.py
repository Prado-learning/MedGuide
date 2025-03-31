from flask import Flask, request, jsonify
import os
import json
from openai import OpenAI

app = Flask(__name__)
app.json.ensure_ascii = False

# 初始化阿里云 QwQ 客户端
client = OpenAI(
    api_key="sk-1d9f630ab5a34072b30ac4630021a643",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

model_name = "qwq-plus"

def call_model(prompt):
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        answer_content = ""
        is_answering = False
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    print(delta.reasoning_content, end='', flush=True)
                    answer_content += delta.reasoning_content
                else:
                    if delta.content and not is_answering:
                        print("\n==================== 完整回复 ====================\n")
                        is_answering = True
                    if delta.content:
                        print(delta.content, end='', flush=True)
                        answer_content += delta.content
        return answer_content.strip() if answer_content else "无法生成引导问题"
    except Exception as e:
        print("[错误] 模型调用失败:", e)
        return "无法生成引导问题"

@app.route('/api', methods=['POST'])
def api_endpoint():
    if request.method == 'POST':
        try:
            raw_data = request.get_data(as_text=True)
            data = json.loads(raw_data)
        except Exception as e:
            print("[错误] 请求 JSON 解码失败:", e)
            return jsonify({"guidance_question": "请求数据格式错误"}), 400

        question = data.get("question", "").strip()
        diagnosis = data.get("diagnosis", "").strip()

        if not question:
            return jsonify({"guidance_question": "未提供有效的患者描述"}), 400

        if diagnosis == "不适用":
            prompt = (
                "你是一名康复医疗领域的专业医师。\n\n"
                "患者提供了如下描述，但当前无法确定诊断：\n"
                f"{question}\n\n"
                "请生成一个或多个有针对性的引导性问题，以帮助更好地判断患者的病情。\n"
                "请直接输出问题内容，不要添加额外的解释。"
            )
            guidance_question = call_model(prompt)
            return jsonify({"guidance_question": guidance_question}), 200

        return jsonify({"guidance_question": "不需要额外引导问题"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
