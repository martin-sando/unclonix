import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
import warnings
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

warnings.simplefilter("ignore")
a = 1000
b = 500

def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """

    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)

    return (kps, features)

def createMatcher(method, crossCheck):
    "Create and return a Matcher Object"

    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)

    # Match descriptors.
    best_matches = bf.match(featuresA, featuresB)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    #print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches


def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    #print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)

        return (matches, H, status)
    else:
        return None

def read_and_prep(path):  # читаем картинку, решейпим в массив
    global a, b
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (a, b), interpolation=cv2.INTER_AREA)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    return img


def kmeans(img, clt):  # применяем метод к-средних
    global a, b
    clt.fit(img)
    new_colors = clt.cluster_centers_[clt.predict(img)]
    img_recolored = new_colors.reshape(b, a, 3)
    clrs = list(new_colors)
    imgc = img_recolored.astype('uint8')
    return clrs, imgc, img_recolored


def make_matrix(clrs, imgc):  # создаем матрицу
    colors, cost1 = [], []  # cost1 - костыль, без которого можно обойтись, но пока что код менять лень
    for x1 in clrs:
        x = list(map(int, x1))
        if str(list(x)) not in cost1:
            colors.append(list(x))
            cost1.append(str(list(x)))

    lat = np.empty((b, a), dtype=np.int)
    for i in range(b):
        for j in range(a):
            lat[i, j] = cost1.index(str(list(imgc[i, j])))
    return lat, colors


def show_matrix(lat, colors, img_recolored):  # выводим матрицу
    plt.figure()
    plt.axis("off")
    plt.imshow(img_recolored.astype('uint8'))
    cmap = LinearSegmentedColormap.from_list('', colors, len(colors))
    # print(colors)
    plt.imshow(lat, cmap=cmap, vmin=0, vmax=len(colors) - 1, alpha=0.4, aspect='auto')
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            plt.text(j, i, lat[i, j], fontsize=12, ha='center', va='center')
    return plt


def calc_sim_index(result):  # считаем индекс сходства (на вход получаем массив матриц, выводим массив индексов)
    all_indexes = []
    for i in range(len(result) - 1):
        for j in range(i + 1, len(result)):
            X, Y = result[i], result[j]
            res = [[X[i][j] - Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]
            res = np.array(res)
            # print(res)
            sim_index = np.count_nonzero(res == 0) / (len(X[0]) * len(X))
            # print(sim_index)
            all_indexes.append(sim_index)
    return all_indexes


def main(a, b, n_clusters, file1, file2):
    clt = KMeans(n_clusters=n_clusters, init='k-means++')
    img = read_and_prep(file1)
    clrs, imgc, img_recolored = kmeans(img, clt)
    lat1, colors1 = make_matrix(clrs, imgc)
    img = read_and_prep(file2)
    clrs, imgc, img_recolored = kmeans(img, clt)
    lat2, colors2 = make_matrix(clrs, imgc)
    return calc_sim_index([lat1, lat2])

cv2.ocl.setUseOpenCL(False)
# select the image id (valid |values 1,2,3, or 4)
feature_extractor = 'brisk' # one of 'sift', 'surf', 'brisk', 'orb'
feature_matching = 'knn'


def check(userlabel, label):
    trainImg = imageio.imread(userlabel)
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)

    queryImg = imageio.imread(label)
    # Opencv defines the color channel in the order BGR.
    # Transform it to RGB to be compatible to matplotlib
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
    ax1.imshow(queryImg, cmap="gray")
    ax1.set_xlabel("", fontsize=14)

    ax2.imshow(trainImg, cmap="gray")
    ax2.set_xlabel("", fontsize=14)

    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)

    # display the keypoints and features detected on both images
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
    ax1.imshow(cv2.drawKeypoints(trainImg_gray,kpsA,None,color=(0,255,0)))
    ax1.set_xlabel("", fontsize=14)
    ax2.imshow(cv2.drawKeypoints(queryImg_gray,kpsB,None,color=(0,255,0)))
    ax2.set_xlabel("", fontsize=14)
    #print("Using: {} feature matcher".format(feature_matching))

    fig = plt.figure(figsize=(20, 8))

    if feature_matching == 'bf':

        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches[:100],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
    #    img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, np.random.choice(matches, 100),
    #                           None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    if M is None:
        #print("Fake (rot)")
        return 0
    (matches, H, status) = M

    # Apply correction
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]


    result = cv2.warpPerspective(trainImg, H,(queryImg.shape[1], queryImg.shape[0]))
    #result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

    plt.figure(figsize=(20,10))
    plt.imsave("rot.jpg", result)


    n_clusters = 4
    file1 = "rot.jpg"
    file2 = label
    si = main(a, b, n_clusters, file1, file2)
    return si[0]








