# import os
from flask import Flask, render_template, request, send_file
import sys
from io import BytesIO
import json
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

img_file = "input1.jpg"
directory = "examples/"

"""-----------------"""
image_width = 300
image_height = 500


def path_file(file):
    return str(file)

def color_detection12(img, n_colors, show_chart=False, output_chart=None):
    # RGB formatına dönüştürme
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
        plt.figure(figsize=[10, 10])
        plt.pie(color_category.values(), labels=color_category.keys(), colors=color_category.keys())
        plt.savefig(output_chart)

    return color_category


def nn(img_file):
    predict = predictor(img_file)
    file = path_file("annotation.csv")
    reader = pd.read_csv(file)
    print(predict)

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
    

    img_array = cv2.imread('examples/grabcut_dress2.png')
    
    #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # cv2.imshow("name",img_array)
    # cv2.waitKey(0)

    color_detection12(img_array, n_colors=3, show_chart=True, output_chart="output22_chart.png")
    # find_dominant_color_quantization(img_array)
nn(img_file)

# @app.route('/upload', methods=['POST'])
# def upload2():
#     if 'img_file' not in request.files:
#         return 'Dosya bulunamadı.'

#     file = request.files['img_file']

#      # Dosyayı belirli bir dizine kaydetme
#     if file:
#          file.save('uploads/' + file.filename)
        
      
#          nn_result = nn('uploads/' + file.filename)
#          color_category = nn_result["color_category"]
        
      
#          plt.figure(figsize=[10, 10])
#          plt.pie(color_category.values(), labels=color_category.keys(), colors=color_category.keys())
        
       
#          buffer = BytesIO()
#          plt.savefig(buffer, format='png')
#          buffer.seek(0)
#          chart_data = buffer.getvalue()
#          buffer.close()

# #         # HTML sayfasında chartı göstermek için base64 formatında veriyi gönderin
#          #chart_image = f"data:image/png;base64,{base64.b64encode(chart_data).decode()}"

# #         # index.html sayfasına dönüş yapın ve chartı göstermek için chart_image'i gönderin
#          return render_template('index.html', chart_image=output_chart)