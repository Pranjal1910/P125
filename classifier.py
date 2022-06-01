import imp
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml('mnist_784' , version = 1, return_X_y = True)    
xtrain , xtest, ytrain, test = train_test_split(X,y ,random_state = 9, train_size = 7500 , test_size = 2500)

xtrainscale = xtrain/255.0
xtestscale = xtest/255.0

clf =  LogisticRegression(solver = 'saga' , multi_class = 'multinomial').fit(xtrainscale, ytrain)
def get_prediction(image):
    impil = Image.open(image)
    imagebw = impil.convert('L')
    imagebwresized = imagebw.resize((28,28), Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(imagebwresized , pixelfilter)
    imagebwresizeinvertedscale = np.clip(imagebwresized - minpixel,0,255)
    maxpixel = np.max(imagebwresized)
    imagebwresizeinvertedscale = np.asanyarray(imagebwresizeinvertedscale)/maxpixel
    testsample = np.array(imagebwresizeinvertedscale).reshape(1,784)
    testpred = clf.predict(testsample)
    return testpred[0]


