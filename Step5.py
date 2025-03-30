from flask import Flask, request, jsonify
import re
import numpy as np
import requests
import json
import os


path = r"./guides/"
with np.load('dia2guide.npz') as data:
    dia2guide = dict(data.items())
model_name = "BUPTmodel-70B"
# url = "http://localhost:8080/v1/chat/completions"
url = "https://9zekn5682505.vicp.fun/v1/chat/completions"
headers = {
    'Content-Type': 'application/json'
}

app = Flask(__name__)
app.json.ensure_ascii = False

@app.route('/api', methods=['POST'])
def api_endpoint():
    if request.method == 'POST':
        data = request.json
        diagnosis = data['diagnosis'] # 获取前文得出的诊断
        guide_input = data['guide_input'] # 获取前文得出的指南
        question = data['question'] # 获取用户输入内容
        if len(guide_input) > 0:
            content = ''
            content = content + 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
            content = content + 'Step 2. 现有一名患者或其家属来咨询，我将给你一份初步诊断与建议作为参考，请根据相关内容对患者的情况进行分析，解答患者问题；\n\n'
            content = content + f'Step 3. 患者描述如下：\n{question}\n\n'
            content = content + f'Step 4. 初步诊断与建议如下：\n{guide_input}\n'
            if diagnosis in dia2guide:
                content = content + f'Step 5. 考虑患者可能的诊断为{diagnosis}，请针对患者实际情况，结合初步诊断与建议中的内容给出具体详细的答复，你的回答要符合康复医疗领域的专业医师的身份，若患者有明确提出问题，请直接对所提问题进行回答。'
            else:
                content = content + f'Step 5. 请针对患者实际情况，结合初步诊断与建议中的内容给出具体详细的答复，你的回答要符合康复医疗领域的专业医师的身份，若患者有明确提出问题，请直接对所提问题进行回答。'
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.3,
                "max_tokens": 1024,
                "top_p": 0.95
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            anwser = data['choices'][0]['message']['content']
        else:
            if diagnosis not in dia2guide:
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": question}],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "top_p": 0.95
                })
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                anwser = data['choices'][0]['message']['content']
            else:
                content = ''
                content = content + 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
                content = content + f'Step 2. 现有一名患者前来咨询，患者描述如下；\n{question}\n\n'
                content = content + f'Step 3. 考虑患者可能的诊断为{diagnosis}，请对患者的情况进行分析，对患者问题进行解答。'
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": content}],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "top_p": 0.95
                })
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                anwser = data['choices'][0]['message']['content']
        return jsonify({"anwser": anwser})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
