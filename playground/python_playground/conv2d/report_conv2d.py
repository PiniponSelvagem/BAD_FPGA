
import numpy as np
def conv2D(image, kernel, bias):    
    # Get image and kernel dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding size
    pad_height = kernel_height // 2
    pad_width  = kernel_width // 2
    
    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')

    # Create output array
    output = np.zeros((image_height, image_width))
    
    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            output[i][j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
            
            # Add bias
            output[i][j] += bias
    
    return output




input = np.array([
    [ 1, 2, 0, 0, 0 ],
    [ 1, 2, 3, 4, 5 ],
    [-1,-2,-3,-4,-5 ],
])
kernel = np.array([ 
    [ 0, 0, 0 ],
    [ 0, 1, 0 ],
    [ 0,-1, 0 ],
])
bias = np.array(1)


output = conv2D(input, kernel, bias)
print(output)
