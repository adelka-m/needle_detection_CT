
from modules import *
from skimage.morphology import erosion 


def find_ransac_needles(img):
    
    needles = needle_detection_3D(img, visualize = False)     

    body_mask = erosion(body_mask_2D(img, closing = False))

    # find needles that are actually pointing into the body and are long enough
    true_needles = []
    m = []
    
    for cand in needles:
        
        paramsransac = cand[1]
        cand = cand[0]
        
        p = in_body_test(cand, body_mask, threshold = 10)

        if p:
            mx_needle_break =  max(np.sum((np.abs(cand[1:] - cand[:-1])),axis = 1)) 
            armx_needle_break =  np.argmax(np.sum((np.abs(cand[1:] - cand[:-1])),axis = 1)) 

            if mx_needle_break < 200:
                # diry jsou jenom maly
                true_needles.append([paramsransac,cand])
            elif armx_needle_break > 200:
                # dira nepatri uz k needle
                true_needles.append([paramsransac, cand[:armx_needle_break]]) 
            elif len(cand) - armx_needle_break > 200:
                # dira nepatri uz k needle
                true_needles.append([paramsransac, cand[armx_needle_break+1:]]) 

    return(true_needles)


def predict_needles(img, dim = (256,256), thr = -200, thrmax = 800):
    
    prediction = np.zeros((np.shape(img)[0],*dim))

    for i in range(np.shape(img)[0]):
        im = img[i,].copy()
        if dim != (512,512):
            im = resize(im, dim, anti_aliasing=True)
        im0 = im.copy()

        if thr != None:
            im[im<thr] = thr

        if thrmax != None:
            im[im>thrmax] = thrmax    

        im = (im - np.min(im)) / np.max(im  - np.min(im))   * 255.0

        data = np.ones((*dim, 3))
        data[:,:,0] = np.squeeze(im) 
        data[:,:,1] = np.squeeze(im)
        data[:,:,2] = np.squeeze(im)

        X = np.reshape(data, (1, *dim, 3))

        prediction[i,] = np.squeeze(model.predict(X))
        
    return(prediction)







