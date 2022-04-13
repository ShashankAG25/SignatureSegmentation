import cv2

# Loading the image
dImage = cv2.imread('./SPODS_Dataset/image (2).png')
# cv2.namedWindow('Show', cv2.WINDOW_NORMAL)
# cv2.imshow("Show" ,dImage)
# cv2.waitKey(0)

# converting to Greyscale
grey_dImage = cv2.cvtColor(dImage, cv2.COLOR_RGB2GRAY)

#Gaussian Blurring
gaussian_dImage = cv2.GaussianBlur(grey_dImage,(15,15),0)

highPass_dImage = cv2.subtract(grey_dImage,gaussian_dImage)
# cv2.namedWindow('Show', cv2.WINDOW_NORMAL)
# cv2.imshow("Show" ,highPass_dImage)
# cv2.waitKey(0)

Y_dImage = cv2.absdiff(grey_dImage,highPass_dImage)
cv2.namedWindow('Show', cv2.WINDOW_NORMAL)
cv2.imshow("Show" ,Y_dImage)
cv2.waitKey(0)
