import cv2
import matplotlib.pyplot as plt


image = cv2.imread('bridge.jpg')
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eql_grayscale_image = cv2.equalizeHist(grayscale_image)

plt.imshow(eql_grayscale_image, cmap='gray')
plt.show() #equalized_image


#graph of the equalized image
histogram = cv2.calcHist([eql_grayscale_image], [0], None, [256], [0, 256])
plt.plot(histogram, color='k')
plt.xlim([0, 256])
plt.show()
