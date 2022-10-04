
from flask import Flask, render_template, jsonify, Response, request
import cv2
import os
import numpy as np
import pandas as pd
import json
# http://0.0.0.0:5001/thumbnailimg?index=0


# app = Flask(__name__, template_folder='templates', static_folder='static')
app = Flask(__name__, template_folder='templates')


@app.route('/thumbnailimg')
def thumbnailimg():
    print("load_iddoc")
    pagefile = []
    index = int(request.args.get('index'))
    if index == None:
        index = 0

    imgperindex = 100
    
    imgpath = request.args.get('imgpath') + "/"
    pagefile = []
    filelist = os.listdir(imgpath)
    if len(filelist)-1 > index+imgperindex:
        page_filelist = filelist[index*imgperindex:index*imgperindex+imgperindex]
    else:
        page_filelist = filelist[index*imgperindex:len(filelist)]

    for fname in page_filelist:
        pagefile.append({'imgpath': imgpath, 'fname': fname})

    data = {'num_page': int(len(filelist)/imgperindex)+1, 'pagefile': pagefile}

    return render_template('index_thumb.html', data=data)


@app.route('/get_img')
def get_img():
    # print("get_img")
    fpath = request.args.get('fpath')
    fpath = fpath
    image_name = fpath.split("/")[-1]

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return  Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tostring() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)