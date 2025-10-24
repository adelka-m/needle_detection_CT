
import numpy as np
from skimage.morphology import area_closing, dilation, erosion 



def body_mask_2D(img, eps = 0.01, k = None, thr = -500, closing = True):
    
    structure = np.ones((3, 3), dtype=np.int32)
    
    if len(np.shape(img)) == 3:
        body = np.zeros(np.shape(img))
    else:
        body = np.zeros((1, *np.shape(img)))
        img = np.reshape(img, (1,*np.shape(img)))
        
    for i in range(np.shape(body)[0]):
        binm = img[i].copy()>thr
        #plt.figure()
        #plt.imshow(binm)

        labeled, ncomponents = label(binm, structure)   
        #print(np.unique(labeled))
        labeled_count = [np.sum(labeled == lbl) for lbl in np.unique(labeled)[1:]]
        
        body[i] = labeled == np.argmax(labeled_count)+1
    body = erosion_dilation(body, k)
    if closing:
        body = close_body_mask(body)
    return 


def close_body_mask(body, n = 50000):
    body_c = body.copy()    
    for i in range(np.shape(body_c)[0]):    
        body_c[i] = area_closing(np.pad(body[i],3), n)[3:-3,3:-3]
    return body_c


def erosion_dilation(img, k = None):
    if k is None:
        k = 5
    for i in range(k):
        img = erosion(img)
    for i in range(k):
        img = dilation(img)
    return img


def in_body_test(needle, body, threshold = 10):
    needletr = np.transpose(needle)
    inbody = np.sum(body[needletr[0], needletr[1], needletr[2]])
    if inbody > threshold:
        return True
    else:
        return False