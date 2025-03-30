from flask import Flask, request, jsonify
import re
import numpy as np
import requests
import json
import os

# url = "http://localhost:8080/v1/chat/completions"
url = "https://9zekn5682505.vicp.fun/v1/chat/completions"
headers = {
    'Content-Type': 'application/json'
}
model_name = "BUPTmodel-70B"
app = Flask(__name__)
app.json.ensure_ascii = False

@app.route('/api', methods=['POST'])
def api_endpoint():
    if request.method == 'POST':
        data = request.json # 获取前端传来的数据
        diagnosis = data['diagnosis'] # 获取诊断结果
        question = data['question'] # 获取患者描述
        guide_list = data['guide_list'] # 获取指南列表
        content = ''
        content = content + 'Step 1. 你是一名康复医疗领域的专业医师；\n\n'
        content = content + 'Step 2. 现有一名患者或其家属来咨询，请你根据患者描述，选择合适的诊疗指南用以辅助对其进行诊断与治疗；\n\n'
        content = content + f'Step 3. 患者描述如下：\n{question}\n\n'
        content = content + 'Step 4. 你可以选择的诊疗指南名称如下：\n'
        stcontent = content
        num = 1
        allchoices = []
        for file in guide_list: # 遍历文件夹中的所有文件
            content = content + f'({num}).' + file + '\n'
            num += 1
            if num > 45: # 如果文件数量超过45个，则停止遍历
                if diagnosis == "不适用":
                    content = content + '\nStep 5. 请根据患者的描述，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
                else:
                    content = content + f'\nStep 5. 请根据患者的描述，考虑患者的诊断为{diagnosis}，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": content}],
                    "temperature": 0.0,
                    "max_tokens": 256
                })
                response = requests.post(url, headers=headers, data=payload)
                data = response.json()
                nowchoice = data['choices'][0]['message']['content']
                allchoices = allchoices + nowchoice.split('\n')
                num = 1
                content = stcontent
        if num > 1:
            if diagnosis == "不适用":
                content = content + '\nStep 5. 请根据患者的描述，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
            else:
                content = content + f'\nStep 5. 请根据患者的描述，考虑患者的诊断为{diagnosis}，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": 256
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            nowchoice = data['choices'][0]['message']['content']
            allchoices = allchoices + nowchoice.split('\n')
            num = 1
            content = stcontent
        cleanchoices = []
        for c in allchoices:
            if c in guide_list:
                cleanchoices.append(c)
        allchoices = cleanchoices
        if len(allchoices) > 1:
            content = stcontent
            num = 1
            for dia in allchoices:
                content = content + f'({num}).' + dia + '\n'
                num += 1
            if diagnosis == "不适用":
                content = content + '\nStep 5. 请根据患者的描述，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
            else:
                content = content + f'\nStep 5. 请根据患者的描述，考虑患者的诊断为{diagnosis}，在所给的诊疗指南范围内选出合适的一篇指南辅助医生对患者进行诊断与治疗，如果有合适的指南，直接输出对应诊疗指南的名称，无需添加其他内容，若有多本指南均可用作参考，请选择一本最为合适，直接输出其对应的名称，如果没有合适的指南，请输出“不适用”。'
            payload = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": 256
            })
            response = requests.post(url, headers=headers, data=payload)
            data = response.json()
            nowchoice = data['choices'][0]['message']['content']
            nowchoice = re.sub(r'\([0-9]+\)\.', '', nowchoice).split('\n')[0]
            if nowchoice not in guide_list:
                nowchoice = allchoices[0]
        elif len(allchoices) > 0:
            nowchoice = allchoices[0]
        else:
            nowchoice = "不适用"
        return jsonify({"question": question, "diagnosis": diagnosis, "guide": nowchoice})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
