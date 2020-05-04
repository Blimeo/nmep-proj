import flask as fsk
import base64
import re
import cv2
import imageio
import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(os.path.dirname(currentdir))
#sys.path.insert(0,parentdir)
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utils import *
from models import *


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
    sel_cls = int(data['cls'])
    base64_data = re.sub('^data:image/.+;base64,', '', raw_image_data)
    png_data = base64.b64decode(base64_data)
    with open("original.png", "wb") as fh:
        fh.write(png_data)
    new_img = remove_transparency('original.png')
    new_img = cv2.resize(new_img, (32, 32), interpolation = cv2.INTER_AREA)
    cv2.imwrite('output.png', new_img)

    # olivers edits - make sure this doesn't break anything
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet('18').to(device)
    model.load_state_dict(torch.load('model', map_location=device))
    model.eval()
    image = torch.Tensor(imageio.imread('output.png')).permute(2, 0, 1).to(device) / 255.
    label = torch.Tensor([sel_cls]).to(device).long()
    eps = 3500
    steps = 40
    # adv is the shifted image as a pytorch tensor which is (1, 3, 32, 32)
    adv = pgd_attack(image.view(3,32,32), sel_cls, model, stepsize=2.5 * eps / steps, eps=eps, steps=steps, constraint='l_2').cpu()
    # upscale to 480 x 480
    upsample = transforms.Compose([transforms.ToPILImage(), transforms.Resize(480), transforms.ToTensor()])
    save_image(upsample(adv), 'converted.png')
    final_img = cv2.imread('converted.png')
    b64_data = base64.b64encode(cv2.imencode('.png', final_img)[1]).decode()
    return b64_data

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

