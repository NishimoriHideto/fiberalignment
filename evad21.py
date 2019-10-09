# coding: utf-8
#import section
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import sys

#read image
argvs = sys.argv
argc = len(argvs)

if (argc != 2):
    print ('Usage: # python {0} [image]'.format(argvs[0]))
    quit()
infile = argvs[1]

img = cv2.imread(infile, 0)

# image operation
nimg = img / np.amax(img)*255
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
ms = 20*np.log(np.abs(fshift)) #magnitude spectrum
msn = ms / np.amax(ms)*255 #normalized magnitude spectrum
ms8=msn.astype(np.uint8)
ms8dn=cv2.fastNlMeansDenoising(ms8,None,7,7,21)
msdnb = ((ms8dn > 127)*255).astype(np.uint8)
kernel = np.ones((5,5), np.uint8)
msdnbop = cv2.morphologyEx(msdnb, cv2.MORPH_OPEN, kernel)
msdnboc = cv2.morphologyEx(msdnbop, cv2.MORPH_CLOSE, kernel)

plt.figure();plt.imshow(msdnb, cmap=plt.cm.gray);plt.title('msdnb')
plt.figure();plt.imshow(msdnbop, cmap=plt.cm.gray);plt.title('msdnbop')
plt.figure();plt.imshow(msdnboc, cmap=plt.cm.gray);plt.title('msdnboc')
plt.show()

ret,thresh = cv2.threshold(msdnboc,127,255,cv2.THRESH_BINARY)
plt.figure();plt.imshow(thresh, cmap=plt.cm.gray);plt.title('thresh')
plt.show()

cnt, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ms8dn, contours, -1, (0,0,255), 3)
#cntdn = cv2.morphologyEx(cnt, cv2.MORPH_OPEN, kernel)

#open_img=ndimage.binary_opening(ms8dnb)
#close_img=ndimage.binary_closing(open_img)
#plt.figure();plt.imshow(open_img);plt.title('open')
#plt.figure();plt.imshow(close_img);plt.title('close')
#plt.show()

#ms0=ms8dn;ms0 = ms0 - ms0
plt.figure();plt.imshow(ms8,cmap=plt.cm.gray);plt.title('ms8')
plt.figure();plt.imshow(msdnb,cmap=plt.cm.gray);plt.title('ms8dnb')
plt.figure();plt.imshow(thresh,cmap=plt.cm.gray);plt.title('threshold')
plt.figure();plt.imshow(cnt,cmap=plt.cm.gray);plt.title('cnt')
plt.show()

plt.figure();plt.imshow(nimg,cmap=plt.cm.gray);plt.title('original')
plt.savefig('original.png')
plt.figure();plt.imshow(ms8dn, cmap=plt.cm.gray);plt.title('fft')
plt.savefig('fft.png')
plt.figure();plt.imshow(cnt, cmap=plt.cm.gray);plt.title('contour')
plt.savefig('contour.png')
plt.show()


plt.figure()
plt.imshow(ms8dn, cmap=plt.cm.gray)
plt.imshow(cnt, cmap=plt.cm.gray, alpha=0.4)
plt.title('fft+contour')
plt.savefig('fft_contour.png')
plt.show()

#msf32=ms8dnb.astype(np.float32)
#plt.imshow(msf32);plt.show()

cntsq = np.vstack(contours).squeeze()
#cntsqdn = cv2.morphologyEx(cntsqdn, cv2.MORPH_OPEN, kernel)
#plt.plot(cntsq.tolist())
#cntsq_splbgr =cv2.split(cntsq)
#cntsq_mrgbgr = cv2.merge((cntsq_splbgr[0],cntsq_splbgr[1],cntsq_splbgr[3]))
#cntsqdn = cv2.fastNlMeansDenoising(cntsq_splbgr,None,5,7,21)
ellipse = cv2.fitEllipse(cntsq)
print(ellipse)
print(ellipse[1][1]/ellipse[1][0])

#plt.figure();plt.imshow(cnt, cmap=plt.cm.gray);plt.title('contour')

im = cv2.ellipse(cnt, ellipse, (127,127,127), 2)

plt.figure();plt.imshow(im, cmap=plt.cm.gray);plt.title('contour+ellipse')
plt.savefig('contour_ellipse.png')
plt.show()
