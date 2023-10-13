from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def find_circles(img_cv):
  img_cv = cv.cvtColor(img_cv, cv.COLOR_GRAY2BGR)
  img_cv = cv.applyColorMap(img_cv, cv.COLORMAP_JET)

  if circles is not None:
    circles = np.uint16(np.around(circles))
    print(circles.shape)
    for i in circles[0,:]:
      # draw the outer circle
      cv.circle(img_cv,(i[0],i[1]),i[2],(0,255,0),2)
      # draw the center of the circle
      cv.circle(img_cv,(i[0],i[1]),2,(0,0,255),3)
    #cv.imwrite('test.png',img_cv)
  cv.imwrite('test.png',img_cv)
  circles = cv.HoughCircles(img_cv, cv.HOUGH_GRADIENT, 3, 10, param1=50, param2=30, minRadius=1, maxRadius= 5)

def plot_preprocessed_img(img_cv_in):
  img_cv = cv.cvtColor(img_cv_in, cv.COLOR_BGR2GRAY)
  #img_cv = cv.GaussianBlur(img_cv,(3,3), 0)
  #ret, thresh = cv.threshold(img_cv, 10, 256,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
  ret, thresh = cv.threshold(img_cv, 115, 256,cv.THRESH_BINARY)
  edge = cv.Canny(thresh, 120, 190, 3)
  cv.imwrite(f"plots/edge/edge_img_lo_120_hi_190_grad_3.png",edge)
  '''
  for i in range(10, 300,5):
    ret, thresh = cv.threshold(img_cv, i, 256,cv.THRESH_BINARY)
    #ret, thresh = cv.threshold(img_cv, i, 256,cv.THRESH_OTSU)
    #ret, thresh = cv.threshold(img_cv, i, 256,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imwrite(f"plots/thresh/thresh_{i}.png",thresh)
  for j in range(100, 200, 20): 
    for k in range(60, 100,10):
      for l in range(2, 8, 1):
        #edge = cv.Canny(img_cv, j, j+k, l)
        edge = cv.Canny(thresh, j, j+k, l)
        cv.imwrite(f"plots/edge/edge_img_lo_{j}_hi_{j+k}_grad_{l}.png",edge)
        edge = cv.GaussianBlur(edge,(3,3), 0)
        cv.imwrite(f"plots/edge/edge_blurred_img_lo_{j}_hi_{j+k}_grad_{l}.png",edge)

  '''

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot
def find_rectangles(img_cv_in):
  #i = 95
  for i in range(140, 200, 10):
    img_cv = cv.cvtColor(img_cv_in, cv.COLOR_BGR2GRAY)
    #img_cv = img_cv_in
    cv.imwrite("bin_img.png",img_cv)
    #ret, thresh = cv.threshold(img_cv, i, 256,cv.THRESH_OTSU)
    #ret, thresh = cv.threshold(img_cv, i, 256,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    img_cv = cv.GaussianBlur(img_cv,(5,5), 0)
    ret, thresh = cv.threshold(img_cv, i, 256,cv.THRESH_BINARY)
    #ret, thresh = cv.threshold(img_cv, i, 256,cv.THRESH_BINARY)
    #ret, thresh = cv.threshold(img_cv, 105, 256,cv.THRESH_BINARY)
    #thresh = cv.GaussianBlur(thresh,(5,5), 0)
    #edge = cv.Canny(thresh, 120, 190, 3)
    #edge = cv.Canny(img_cv, 100, 150, 2)
    cv.imwrite(f"plots/thresh/thresh_img_{i}.png",thresh)
    #cv.imwrite(f"plots/edge/edge_img_{i}.png",edge)
 
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print("Number of contours detected: ", len(contours))

    img_cv2 = cv.imread('Lato_SUD_TIR.tif', cv.IMREAD_UNCHANGED)
    print(img_cv2.shape[0])
    print(img_cv.shape[0])
    img_cv2 = cv.resize(img_cv2, (img_cv.shape[1], img_cv.shape[0]), interpolation = cv.INTER_AREA)
    img_cv2 = cv.cvtColor(np.uint8(img_cv2), cv.COLOR_GRAY2BGR)
    img_cv2 = cv.applyColorMap(img_cv2, cv.COLORMAP_TURBO)
    #img_cv2 = cv.applyColorMap(img_cv2, cv.COLORMAP_JET)

    w_arr = []
    h_arr = []
    for k,cnt in enumerate(contours):
      x1, y1 = cnt[0][0]
      approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
      if len(approx) == 4:
        #x, y, w, h = cv.boundingRect(cnt)
        '''
        img_crop, img_rot = crop_rect(img_cv2,rect)
        print(type(img_crop))
        if not(type(img_crop) != 'NoneType'):
          print("here")
          cv.imwrite(f"plots/crops/cropped_rect_{k}_param_{i}.png",img_crop)
        '''
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        x1 = box[0]
        x2 = box[1]
        x3 = box[2]
        w = np.sum(np.power(x2-x1,2))
        h = np.sum(np.power(x3-x1,2))
        #print(f"box: {box}")
        if( w >30**2 and h > 30**2 and w < 120**2 and h<120**2):
          print(f"width: {w}, height: {h}")
          w_arr.append(w)
          h_arr.append(h)
          img_cv = cv.drawContours(img_cv, [cnt], -1, (0, 255,0),8)
          #img_cv2 = cv.drawContours(img_cv2, [cnt], -1, (255, 255,0),8)
          #img_cv2 = cv.drawContours(img_cv2, [box], -1, (255, 255,0),8)
          cv.putText(img_cv, f"{k}", (x1[0], x1[1]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255,0), 2)

    fig, axs =  plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(w_arr, bins=50)
    axs[1].hist(h_arr, bins=50)
    plt.savefig(f"plots/hist/hist_rec_{i}.png")
    cv.imwrite(f"plots/contours/test_rec_{i}.png",img_cv)
    #cv.imwrite(f"plots/contours/test_rec_{i}.png",img_cv2)
    #cv.imshow("Shapes", img_cv)
    #cv.waitKey(0)
    cv.destroyAllWindows()

def main():
  Image.MAX_IMAGE_PIXELS = 254162108
  #im = Image.open('Lato_SUD_LERS.tif')
  #im = Image.open('Lato_SUD.tif')
  im = Image.open('Ortho1mw_Lres.tif')
  im_array = np.array(im)
  #im_array = im_array * (im_array > 30)
  print('mean')
  print(im_array.mean())
  print('max')
  print(im_array.max())
  print('min')
  print(im_array.mean())
  print(im.getbands())
  print(im.tell())
  print(im_array.shape)
  print(im_array.size)
  img_cv = im_array.astype(np.uint8)

  find_rectangles(img_cv)
  #plot_preprocessed_img(img_cv)
  #find_circle(img_cv)



if __name__ == '__main__':
  main()
