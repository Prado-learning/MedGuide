# Step1API.py - 使用阿里QwQ模型替换（增强prompt + 输出候选诊断 + 修复中文乱码）
from flask import Flask, request, jsonify, Response
import re
import numpy as np
from openai import OpenAI
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

path = r"./guides/"
with np.load('dia2guide.npz') as data:
    dia2guide = dict(data.items())

diagnosis_list = list(dia2guide.keys())
print("现有诊断列表:", diagnosis_list)

client = OpenAI(
    api_key="sk-1d9f630ab5a34072b30ac4630021a643",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

model_name = "qwq-plus"
CHUNK_SIZE = 10
MAX_WORKERS = 5

def call_model(prompt):
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        end = time.time()
        print(f"请求耗时: {end - start:.2f}秒")

        answer_content = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content
        return answer_content.strip() if answer_content else "不适用"
    except Exception as e:
        print("[错误] 请求或解析出现异常:", e)
        import traceback
        traceback.print_exc()
        return f"[模型调用失败] {str(e)}"

@app.route('/api', methods=['POST'])
def api_endpoint():
    if request.method == 'POST':
        data = request.json
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"question": question, "diagnosis": "不适用"}), 200

        chunks = [diagnosis_list[i:i+CHUNK_SIZE] for i in range(0, len(diagnosis_list), CHUNK_SIZE)]

        def process_chunk(chunk):
            prompt = (
                "你是一名康复医疗领域的专业医师。\n"
                "请结合患者提供的主诉、症状表现、体征和既往史，从下列诊断中合理推理选择最相关的一个诊断；如果都不符合则输出“不适用”。\n"
                "请特别关注是否有震颤、步态异常、偏瘫、动作迟缓、既往脑梗史等关键特征。\n"
                f"患者描述如下：\n{question}\n"
                "诊断列表：\n"
            )
            for idx, dia in enumerate(chunk, 1):
                prompt += f"({idx}). {dia}\n"
            prompt += "\n请直接输出最可能诊断(或'不适用')，不要多余解释。"
            result_text = call_model(prompt)
            result_text = re.sub(r"^\(\d+\)\.\s*", "", result_text)
            pick = result_text.strip().split("\n")[0].strip()
            print(f"\n诊断结果: {pick}")
            return pick if pick in chunk else "不适用"

        candidate_diagnoses = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {executor.submit(process_chunk, c): c for c in chunks}
            for future in as_completed(future_to_chunk):
                pick = future.result()
                if pick != "不适用":
                    candidate_diagnoses.append(pick)

        print("候选诊断：", candidate_diagnoses)

        if not candidate_diagnoses:
            return jsonify({"question": question, "diagnosis": "不适用", "candidates": []}), 200

        if len(candidate_diagnoses) == 1:
            return jsonify({"question": question, "diagnosis": candidate_diagnoses[0], "candidates": candidate_diagnoses}), 200

        final_prompt = (
            "你是一名康复医疗领域的专业医师。\n"
            f"患者描述如下：{question}\n"
            "已从不同分组中筛出了以下候选诊断：\n"
        )
        for idx, d in enumerate(candidate_diagnoses, 1):
            final_prompt += f"({idx}). {d}\n"
        final_prompt += (
            "\n请从这些候选中选一个最适合的诊断，如果都不适用则输出'不适用'。只输出最终诊断名称即可。"
        )
        final_text = call_model(final_prompt)
        final_text = re.sub(r"^\(\d+\)\.\s*", "", final_text)
        final_pick = final_text.strip().split("\n")[0].strip()
        print(f"\n最终诊断: {final_pick}")
        return jsonify({
            "question": question,
            "diagnosis": final_pick if final_pick in candidate_diagnoses else "不适用",
            "candidates": candidate_diagnoses
        }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
