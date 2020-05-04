import flask as fsk
import base64
import re
import cv2

app = fsk.Flask(__name__)

def remove_transparency(img):
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    trans_mask = image[:,:,3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

@app.route('/query', methods = ['POST'])
def get_query_from_react():
    data = fsk.request.get_json()
    raw_image_data = data['data']
    base64_data = re.sub('^data:image/.+;base64,', '', raw_image_data)
    png_data = base64.b64decode(base64_data)
    with open("original.png", "wb") as fh:
        fh.write(png_data)
    new_img = remove_transparency('original.png')
    new_img = cv2.resize(new_img, (32, 32), interpolation = cv2.INTER_AREA)
    cv2.imwrite('output.png', new_img)
    return raw_image_data

if __name__ == '__main__':
    app.run()
