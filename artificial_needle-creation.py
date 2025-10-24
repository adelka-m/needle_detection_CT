
from modules import *


df = pd.read_csv('YOURPATH/datapaths.csv')


k = 0
for path in df.Path:
    for scans in df[df['Path'] == path].Scans:
        for scan in scans:
            img = load_folder(path_01 + path + '/' + scan)
            print(path + '/' + scan, np.shape(img)[0])
            if np.shape(img)[0] > 15 and np.shape(img)[0] < 90:
                
                sliceImg = img[np.shape(img)[0]//2].copy()
                helpImg = sliceImg.copy()
                body = body_mask(helpImg)
                helpImg = helpImg * body
                helpImg = erosion(erosion(erosion(erosion((helpImg > 30) * (helpImg < 200)))))
                helpImg = close_body_mask(helpImg, n = 5000)
                helpImg= np.squeeze(helpImg)
                
                structure = np.ones((3, 3), dtype=np.int32)
                labeled, ncomponents = label(helpImg, structure) 
                labeled_count = [np.sum(labeled == lbl) for lbl in np.unique(labeled)[1:]]
                biggest = labeled == np.argmax(labeled_count)+1
              
                if np.max(np.mean(np.nonzero(biggest), axis =1)) < 350:
                    skin = np.zeros_like(body)
                    skin[:,:300,:300] = sobel(body)[:,:300,:300]
                    skinPts = np.transpose(np.nonzero(np.squeeze(skin)))[::4]
                    liverPts = np.transpose(np.nonzero(np.squeeze(biggest)))
                    
                    insertPtIDs = np.random.choice(np.shape(skinPts)[0], 
                                       np.random.randint(4,18))
                    
                    needleList = []

                    for index in insertPtIDs:
                        insertPt = skinPts[index]
                        targetPt = [512,512]
                        while targetPt[1] > 250:
                            targetPtID = np.random.choice(np.shape(liverPts)[0], 1)
                            targetPt = liverPts[targetPtID][0]
                            length = np.linalg.norm(insertPt - targetPt)                            
                            if length > 150:
                                targetPt = [512,512]
                                
                        if np.abs(targetPt[0] - insertPt[0]) > np.abs(targetPt[1] - insertPt[1]):
                            if targetPt[0] < insertPt[0]:
                                m = (targetPt[1]-insertPt[1])/(targetPt[0]-insertPt[0])
                                b = (targetPt[0]*insertPt[1] - insertPt[0]*targetPt[1])/(targetPt[0]-insertPt[0])
                                
                                xx = np.linspace(targetPt[0],insertPt[0], dtype = int)
                            else:
                                m = -(-targetPt[1]+insertPt[1])/(targetPt[0]-insertPt[0])
                                b = -(-targetPt[0]*insertPt[1] + insertPt[0]*targetPt[1])/(targetPt[0]-insertPt[0])
                                
                                xx = np.linspace(insertPt[0],targetPt[0], dtype = int)
                            
                            yy = (m * xx + b).astype(int)
                            
                        else:
                            if targetPt[1] < insertPt[1]:
                                m = (targetPt[0]-insertPt[0])/(targetPt[1]-insertPt[1])
                                b = (targetPt[1]*insertPt[0] - insertPt[1]*targetPt[0])/(targetPt[1]-insertPt[1])
                                yy = np.linspace(targetPt[1], insertPt[1], dtype = int)
                                 
                            else:
                                m = -(-targetPt[0]+insertPt[0])/(targetPt[1]-insertPt[1])
                                b = -(-targetPt[1]*insertPt[0] + insertPt[1]*targetPt[0])/(targetPt[1]-insertPt[1])
                                yy = np.linspace(insertPt[1], targetPt[1], dtype = int)
                                
                            xx = (m * yy + b).astype(int)
                        

                        if (np.max(gaussian(sliceImg[xx,yy])) < 200 and np.min(gaussian(sliceImg[xx,yy])) > -400):
                            
                            needleImg = np.zeros_like(sliceImg)
                            needleImg[xx,yy] = 1
                            needleImg = dilation(needleImg)
                            needleList.append(np.nonzero(needleImg))
                            sliceImg = sliceImg + 200*(needleImg)

                
                np.save('SAVEFOLDER/scan_needles_' + str(k).zfill(3) + '.npy', sliceImg)
                np.save('SAVEFOLDER/needles_' + str(k).zfill(3) + '.npy', needleList, allow_pickle=True)
                k += 1
                
                