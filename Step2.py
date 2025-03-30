# 返回对应的值
from flask import Flask, request, jsonify
import numpy as np
import os


with np.load('guide2num.npz') as data:
    guide2num = dict(data.items())
filenames = list(guide2num.keys())
with np.load('dia2guide.npz') as data:
    dia2guide = dict(data.items())

app = Flask(__name__)
app.json.ensure_ascii = False

@app.route('/api', methods=['POST'])
def api_endpoint():
    if request.method == 'POST':
        data = request.json
        diagnosis = data['diagnosis']
        question = data['question']
        if diagnosis in dia2guide:
            if len(dia2guide[diagnosis]) > 0:
                # 若诊断在诊断-指南表中且指南列表中包含指南，则返回该诊断对应的指南列表
                return jsonify({"question": question, "diagnosis": diagnosis, "guide_list": list(dia2guide[diagnosis])})
        # 否则返回所有指南名称构成的列表
        return jsonify({"question": question, "diagnosis": diagnosis, "guide_list": filenames})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
