import cv2
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

'''
Example showing how to build a Sobel filter for convolution using 
locally defined constant kernel.
'''

def main():
    # setup OpenCL
    platforms = cl.get_platforms()  # a platform corresponds to a driver (e.g. AMD)
    platform = platforms[0]  # take first platform
    devices = platform.get_devices(cl.device_type.GPU)  # get GPU devices of selected platform
    device = devices[0]  # take first GPU
    context = cl.Context([device])  # put selected GPU into context object
    queue = cl.CommandQueue(context, device)  # create command queue for selected GPU and context

    # read image
    imgIn = cv2.imread('photographer.png', cv2.IMREAD_GRAYSCALE)

    # get shape of input image, allocate memory for output to which result can be copied to
    shape = imgIn.T.shape
    imgOut = np.empty_like(imgIn)

    # create image buffers which hold images for OpenCL
    imgInBuf = cl.image_from_array(context, ary=imgIn, mode="r", norm_int=True, num_channels=1)
    imgOutBuf = cl.image_from_array(context, ary=imgOut, mode="w", norm_int=True, num_channels=1)

    # load and compile OpenCL program
    program = cl.Program(context, open('sobel_convolution_kernel_code.cl').read()).build()

    # run convolution kernel
    program.custom_convolution_2d(queue, shape, None, imgInBuf, imgOutBuf)

    # copy back output buffer
    cl.enqueue_copy(queue, imgOut, imgOutBuf, origin=(0, 0), region=shape,
                    is_blocking=True)  # wait until finished copying resulting image back from GPU to CPU

    # save imgOut
    cv2.imwrite('photographer_convolved.png', imgOut)

    # show images
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(imgIn, cmap='gray')
    ax[1].imshow(imgOut, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
