from flask import Flask, render_template, request, send_file
import torch
import prmetest4 as pm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io

matplotlib.use('agg')

def matshow(x):
   # 계산
   re_x = x
   re_y = round(float(torch.sigmoid(re_x * pm.W + pm.b)), 2)
   plt.scatter(re_x, re_y, c='blue', s=100, label='Predicted Point', zorder=3)  # 새로운 점을 파란색으로 표시

   # x_train 값 시각화
   x_train_np = pm.x_train.numpy()
   y_train_np = pm.y_train.numpy()
   plt.scatter(x_train_np[:, 0], y_train_np[:], c=pm.y_data, s=100, cmap='viridis', zorder=1)

   # 시그모이드 함수 시각화 (단일 입력 변수 x에 대해)
   x_values = np.linspace(-10, 10, 100)
   z = pm.W[0][0].item() * x_values + pm.b.item()
   sigmoid = 1 / (1 + np.exp(-z))

   plt.plot(x_values, sigmoid, label='Sigmoid Function', zorder=2)
   plt.title('Sigmoid Function')
   plt.xlabel('x')
   plt.ylabel('Sigmoid(x)')
   plt.xlim(-2, 5)
   plt.legend()

   # 이미지 메모리에 저장
   img = io.BytesIO()
   plt.savefig(img, format='png')
   img.seek(0)
   plt.close()
   
   return img

app = Flask(__name__, template_folder='../')

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/result', methods=['POST'])
def page():
   value = torch.tensor([float(request.form['inputValue'])])
   
   # 계산 수행
   cal = round(abs(float(torch.sigmoid(value * pm.W + pm.b)) * 100 -0.01), 2)
   res = f"crps: {cal}%"

   return render_template('result.html', diagnosis_result=res, image_file='resultImg.png')

@app.route('/resultImg.png')
def result_img():
   value = float(request.args.get('value'))
   img = matshow(value)
   return send_file(img, mimetype='image/png')

if __name__ == '__main__':
   app.run('0.0.0.0', port=5000, debug=True)

