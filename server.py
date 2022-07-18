# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, render_template, Response
from predict import load_model_weights, predict
import os
from werkzeug.utils import secure_filename
import cv2
import requests

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG', 'bmp', 'jpeg'])
model = load_model_weights('./output/best.h5')


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        flag, frame = self.video.read()
        ret, frame_encode = cv2.imencode('.jpg', frame)
        return frame_encode.tobytes()

    def get_img(self):
        flag, frame = self.video.read()
        return frame


video_cam = VideoCamera()


def video_flow(cam):
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def upload_get_result(img_url):
    files = {"file": ("img_url" + ".png", open(img_url, 'rb'), "iamge/png")}
    r = requests.post('http://127.0.0.1:10086/predict', files=files)
    return r.json()['result']
    # return r


def if_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST', 'GET'])
def get_classify_res():
    # 所有上传的文件都会被保存在 ./static/images/ 下
    if request.method == 'POST':
        f = request.files['file']
        if not (f and if_allowed(f.filename)):
            return jsonify({"error": str(1001),
                            "message": "请检查文件后缀，支持的后缀有 png, PNG, jpg, JPG, bmp, jpeg"})
        base_path = os.path.dirname(__file__)
        upload_path = os.path.join(base_path, 'static/images', secure_filename(f.filename))
        f.save(upload_path)
        try:
            pred_label, pred_msg = predict(model, upload_path)
            response = jsonify({"result": pred_msg})
            return response
        except Exception as e:
            print('running in to error:', e)
            return jsonify({"error": str(1002)})
    return render_template('predict.html')


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_flow(video_cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/shot_photo')
def shot_photo():
    frame = video_cam.get_img()
    if (os.path.exists('./test.png')):
        os.remove('./test.png')
    cv2.imwrite('./test.png', frame)
    res = upload_get_result('./test.png')
    return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10086)
    # print(model.summary())
    # print(os.path.dirname(__file__))
    # pred_label, pred_msg = predict(model, './static/images/xbox_controller.jpeg')
    # print(pred_msg)

