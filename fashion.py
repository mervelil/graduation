
from flask import Flask, render_template, request, send_file,jsonify, url_for
import sys
from io import BytesIO
import json
from flask import send_from_directory
import os
from collections import Counter
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
import cv2
import numpy as np
import pandas as pd

from keras.models import model_from_json
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#output_chart = os.path.join(UPLOAD_FOLDER, 'color_chart.png')
# img_file = "red.jpg"
# directory = "examples/"
output_chart="static/disaktar.png"



@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')

    return response
@app.route('/uploads/<filename>')
def uploaded_file(filename):
      return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
@app.route('/')
def base():
    return render_template('threedimage.html')
@app.route('/index2')
def index2():
    return render_template('index2.html')
@app.route('/deneme')
def deneme():
    return render_template('deneme.html')
@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/uploadem')
def uploadem():
    return render_template('upload.html')  

@app.route('/search')
def search():
    return render_template('search.html')  
@app.route('/index2', methods=['GET', 'POST'])
def upload(): 
    chart_image = None
    if request.method == 'POST':
        if 'img_file' in request.files:
            file = request.files['img_file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                img = predictor(file_path)  
                img_cut = nn(file_path)
                uploaded_image = f'uploads/{filename}'
                if img is not None:
                # color_detection_chart = color_detection12(img_cut, n_colors=3, show_chart=True, output_chart='static/color_chart.png')
                # color_dict = color_detection12(img_cut, n_colors=3, show_chart=True, output_chart=output_chart)
                 chart_image = output_chart
                else:
                    print("RESIM PROBLEMİ")
                    pass

    return render_template('index2.html', chart_image=chart_image,uploaded_image=uploaded_image)

def predictor(img_file):
    img = cv2.imread(img_file)
    resize = cv2.resize(img, (64, 64))
    # resize = np.expand_dims(resize, axis=0)

    img_fin = np.reshape(resize, [1, 64, 64, 3])
    json_file = open('model/binaryfas10.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/binaryfashion.h5")
    # print("Loaded model from disk")

    prediction = loaded_model.predict(img_fin)

    prediction = np.squeeze(prediction, axis=1)
    predict = np.squeeze(prediction, axis=0)
    return int(predict)


"""Neural Network Decoding"""
""" The coordinates are created and trained"""
"""-----------------"""
image_width = 300
image_height = 500


def path_file(file):
    return str(file)




import cv2
import numpy as np



def color_detection12(img, n_colors, show_chart=False, output_chart=None):
    # RGB formatına dönüştürme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   # print(img.shape)
    # Alfa kanalını kaldırma
    img = img[:, :, :3]  # İlk üç kanalı (RGB) al

    clf = KMeans(n_clusters=n_colors)
    colors = clf.fit_predict(img.reshape(-1, 3))
    counts = Counter(colors)
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in range(n_colors)]
    hex_colors = ['#%02x%02x%02x' % tuple(map(int, color)) for color in ordered_colors]

    color_category = dict(zip(hex_colors, counts.values()))

    if show_chart:
        plt.figure(figsize=[5, 5])
        plt.pie(color_category.values(), labels=color_category.keys(), colors=color_category.keys())
        plt.savefig(output_chart)

    return color_category




def nn(img_file):
    predict = predictor(img_file)
    file = path_file("annotation.csv")
    reader = pd.read_csv(file)
    #print(predict)

    img = cv2.imread(img_file)
    img = cv2.resize(img, (image_width, image_height))
    # seg = img(img, reader.x1[predict], reader.y1[predict], reader.x2[predict], reader.y2[predict], reader.i[predict])

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)

    fgdModel = np.zeros((1, 65), np.float64)

    rect = (reader.x1[predict], reader.y1[predict], reader.x2[predict], reader.y2[predict])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, reader.i[predict], cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img_cut = img*mask2[:, :, np.newaxis]
    

    img_array = cv2.imread(img_file)
    
#     cv2.imshow("name",img_array)
    #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#     cv2.waitKey(0)
    img_chart=color_detection12(img_cut, n_colors=3, show_chart=True, output_chart=output_chart)
    return img_chart
    # find_dominant_color_quantization(img_array)
#nn(file_path)
from flask import Flask, request, render_template
from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage


# Kategori listesi
cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

def fix_channels(t):
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)

def idx_to_text(i):
    return cats[i]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        cl = p.argmax()
        ax.text(xmin, ymin, idx_to_text(cl), fontsize=10, bbox=dict(facecolor=c, alpha=0.8))
    plt.axis('off')
    plt.show()
    plt.savefig("static/image.png")

def visualize_predictions(image, outputs, threshold=0.8):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    plot_results(image, probas[keep], bboxes_scaled)

MODEL_NAME = "valentinafeve/yolos-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
@app.route("/result")
def result():
    return render_template("result.html")

@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream)
            image = fix_channels(ToTensor()(image))
            image = image.resize((600, 800))
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            visualize_predictions(image, outputs, threshold=0.5)
            return render_template('result.html', user_image='static/image.png')
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1',port=5000)

