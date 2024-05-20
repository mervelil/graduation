def find_dominant_color_quantization(img_file, k=3):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV kullanarak RGB formatına dönüştürme

    # Renk ölçeklendirme işlemi uygula
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    dominant_colors = centers[np.bincount(labels.flatten()).argmax()]
    return dominant_colors.tolist()

def rgb2hex(color):
    return '#{:02x}{:02x}{:02x}'.format(
        int(color[0]), int(color[1]), int(color[2])
    )

def color_detection(img, n_colors, show_chart=False, output_chart=None):  # Gets a PNG image with alpha layer.
    img = img[img[:, :, 3] == 255][:, :3]
    
    clf = KMeans(n_clusters=n_colors)

    colors = clf.fit_predict(img)

    counts = Counter(colors)
    print('Counts: ', counts)
    center_colors = clf.cluster_centers_
    print('Center Colors: ', center_colors)

    ordered_colors = [center_colors[i] for i in range(n_colors)]
    hex_colors = [rgb2hex(ordered_colors[i]) for i in range(n_colors)]
    rgb_colors = [ordered_colors[i] for i in range(n_colors)]

    color_category = dict()

    for idx, hex_color in enumerate(hex_colors):
        color_category[hex_color] = counts[idx]

    print('Color Category: ', color_category)
    print('Color Hex Codes: ', hex_colors)
    print('Color RGB: ', rgb_colors)

    if (show_chart):
        plt.figure(figsize=[10, 10])
        plt.pie(color_category.values(), labels=color_category.keys(), colors=color_category.keys())
        plt.savefig(output_chart)

    return color_category

def color_detection2(img, n_colors, show_chart=False, output_chart=None):
    # Convert RGBA image to RGB
    img = Image.fromarray(img.astype('uint8'), 'RGBA')       
    img = img.convert('RGB')
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_array = np.array(img)

    clf = KMeans(n_clusters=n_colors)

    colors = clf.fit_predict(img_array.reshape(-1, 3))

    counts = Counter(colors)
    print('Counts: ', counts)
    center_colors = clf.cluster_centers_
    print('Center Colors: ', center_colors)

    ordered_colors = [center_colors[i] for i in range(n_colors)]
    hex_colors = [rgb2hex(ordered_colors[i]) for i in range(n_colors)]
    rgb_colors = [ordered_colors[i] for i in range(n_colors)]

    color_category = dict()

    for idx, hex_color in enumerate(hex_colors):
        color_category[hex_color] = counts[idx]

    print('Color Category: ', color_category)
    print('Color Hex Codes: ', hex_colors)
    print('Color RGB: ', rgb_colors)

    if (show_chart):
        plt.figure(figsize=[10, 10])
        plt.pie(color_category.values(), labels=color_category.keys(), colors=color_category.keys())
        plt.savefig(output_chart)

    return color_category

def color_detectionsiyahsiz(img, n_colors, show_chart=False, output_chart=None):
    # RGB formatına dönüştürme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Siyah olan pikselleri kaldırma
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([30, 30, 30], dtype=np.uint8)
    mask = cv2.inRange(img, lower_black, upper_black)
    img[mask > 0] = [255, 255, 255]  # Siyah olan pikselleri beyaz yap

    # K-Means ile renk gruplandırma
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
    
    cv2.imshow("name",img_array)
   # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    cv2.waitKey(0)
    color_detection12(img_array, n_colors=3, show_chart=True, output_chart=output_chart)

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


    # find_dominant_color_quantization(img_array)


#nn ici
    # dominant_colors = find_dominant_color_quantization(img_file, k=3)
    # print("Baskın Renkler:", dominant_colors)


    # image = np.array(Image.open(img_file))

    # color_detection(image, n_colors=3, show_chart=True, output_chart=output_chart)
  #  img_file_output = directory + "/" + str(img_file)
  #  cv2.imwrite(img_file_output, img_cut)
    # cv2.imwrite('examples/grabcut_dress2.png', img_cut)
    # img_cut_png = Image.open('examples/grabcut_dress2.png')
    # img_array = np.array(img_cut_png)
    # #img = Image.fromarray(img_array, 'RGBA')
    # color_detection2(np.array(img_cut_png), n_colors=3, show_chart=True, output_chart=output_chart)

