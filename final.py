
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy
from scipy import ndimage


def convolute(image, filter, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(filter.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = filter.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(filter * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= filter.shape[0] * filter.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_Mask(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()

    return kernel_2D


def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_Mask(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolute(image, kernel, average=True, verbose=verbose)

def sobil(image, convert_to_degree=False, verbose=False):
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    new_image_x = convolute(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolute(image, np.flip(filter.T, axis=0), verbose)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    edge_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    edge_magnitude *= 255.0 / edge_magnitude.max()

    if verbose:
        plt.imshow(edge_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    edge_direction = np.arctan2(new_image_y, new_image_x)

    if convert_to_degree:
        edge_direction = np.rad2deg(edge_direction)
        edge_direction += 180

    return edge_magnitude, edge_direction

def Prewitt(image, convert_to_degree=False, verbose=False):
    filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    new_image_x = convolute(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolute(image, np.flip(filter.T, axis=0), verbose)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    edge_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    edge_magnitude *= 255.0 / edge_magnitude.max()

    if verbose:
        plt.imshow(edge_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    edge_direction = np.arctan2(new_image_y, new_image_x)

    if convert_to_degree:
        edge_direction = np.rad2deg(edge_direction)
        edge_direction += 180

    return edge_magnitude, edge_direction



def non_maxima_suppression(gradient_magnitude, gradient_direction):
    verbose=False
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)
    PI = 180
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Non Max Suppression")
        plt.show()

    return output


def double_threshold(image, low, high, weak, verbose=False):
    output = np.zeros(image.shape)

    strong = 255

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("threshold")
        plt.show()

    return output

def firstDerivativeEdgeDetector(image, verbose = False):

    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    new_image_x = convolute(image, filter,verbose)
    plt.imshow(new_image_x, cmap='gray')
    plt.title("1st Deriv --> X")
    plt.show()
    new_image_y = convolute(image, np.flip(filter.T, axis=0),verbose)
    plt.imshow(new_image_y, cmap='gray')
    plt.title("1st Deriv --> Y")
    plt.show()
    edge_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    edge_magnitude *= 255.0 / edge_magnitude.max()
    plt.imshow(edge_magnitude, cmap='gray')
    plt.title("1st Derivative ")
    plt.show()
    return edge_magnitude

def secondDerivativeEdgeDetector(image ,verbose = False):
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    new_image_x = convolute(image, filter, verbose)
    new_image_y = convolute(image, np.flip(filter.T, axis=0), verbose)
    edge_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    edge_magnitude *= 255.0 / edge_magnitude.max()
    plt.imshow(scipy.ndimage.filters.laplace(edge_magnitude, mode = 'reflect'), cmap='gray')
    plt.title(" 2nd Deriv")
    plt.show()
    return edge_magnitude\

def clarify(img, lowRatio, highRatio):
    highThreshold = img.max() * highRatio;
    lowThreshold = highThreshold * lowRatio;
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
# setting the strong and weak values
    weak = np.int32(25)
    strong = np.int32(255)
# setting the strong and non-relevant indexes
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
# setting the weak indeces
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
# fitting values in indeces
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return (res, weak, strong)



def Edge_Linking(image):
    img, weak, strong = clarify(image, 0.05, 0.09)
    M, N = img.shape
    for i in range(1, M - 1):

        for j in range(1, N - 1):

            if (img[i, j] == weak):

                # checking for angle 0 , 45 & 315
                if (img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong):
                    img[i, j] = strong

                # checking for angle 90 & 270
                elif (img[i, j - 1] == strong) or (img[i, j + 1] == strong):
                    img[i, j] = strong

                # checking for angle 180 , 135 & 255
                elif (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong):
                    img[i, j] = strong

                else:
                    img[i, j] = 0

    fig = plt.figure()
    plt.gray()

    y = fig.add_subplot()

    y.imshow(img)
    plt.title("Edge linking")
    plt.show()

    return img

def Canny(image):
    print("-------------------------------- Adding Blurr ----------------------------------------")
    blurred_image = gaussian_blur(image, kernel_size=9, verbose=False)

    print("-------------------------------- 1st Derivative ----------------------------------------")
    firs_deriv_img = firstDerivativeEdgeDetector(blurred_image)

    print("-------------------------------- 2nd Derivative ----------------------------------------")
    secondDerivativeEdgeDetector(blurred_image,verbose=False)

    print("-------------------------------- Prewitt Filter ----------------------------------------")
    edge_magnitude, edge_direction = Prewitt(blurred_image, convert_to_degree=True, verbose=False)

    print("-------------------------------- Sobil Filter ----------------------------------------")
    edge_magnitude, edge_direction = sobil(blurred_image, convert_to_degree=True, verbose=False)

    new_image = non_maxima_suppression(edge_magnitude, edge_direction)
    weak = 50
    new_image = double_threshold(new_image, 5, 20, weak=weak, verbose=False)
    plt.imshow(new_image, cmap='gray')
    plt.title("Canny Edge Detector")
    plt.show()
    Edge_Linking(new_image)

    return edge_magnitude,edge_direction





if __name__ == '__main__':
    image = cv2.imread( r"26.jpg")
    Canny(image)
