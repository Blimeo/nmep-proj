import flask
from flask import request
from flask import jsonify
import base64
import re
import numpy
import cv2

app = flask.Flask(__name__)

@app.route('/query', methods = ['POST'])
def get_query_from_react():
    data = request.get_json()
    raw_image_data = data['data']
    base64_data = re.sub('^data:image/.+;base64,', '', raw_image_data)
    png_data = base64.b64decode(base64_data)
    with open("original.png", "wb") as fh:
        fh.write(png_data)
    image = cv2.imread('original.png', cv2.IMREAD_UNCHANGED)    
    trans_mask = image[:,:,3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    new_img = cv2.resize(new_img, (32, 32), interpolation = cv2.INTER_AREA)
    print(type(new_img))
    cv2.imwrite('output.png', new_img)
    return 'HELLO FROM FLASK BIBBER'

if __name__ == '__main__':
    app.run()
