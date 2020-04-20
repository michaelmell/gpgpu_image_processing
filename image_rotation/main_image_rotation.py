import cv2
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

"""
Here we could use a variable sampler:
    self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)
"""

def main():
    # setup OpenCL
    platforms = cl.get_platforms()  # a platform corresponds to a driver (e.g. AMD, NVidia, Intel)
    platform = platforms[0]  # take first platform
    devices = platform.get_devices(cl.device_type.GPU)  # get GPU devices of selected platform
    device = devices[0]  # take first GPU
    context = cl.Context([device])  # put selected GPU into context object
    queue = cl.CommandQueue(context, device)  # create command queue for selected GPU and context

    # prepare data
    imgIn = cv2.imread('photographer.png', cv2.IMREAD_GRAYSCALE)

    rotation_angle = np.pi/4
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    info = cl.sampler_info

    # sampler = cl.Sampler(context, False, cl.addressing_mode.REPEAT, cl.filter_mode.LINEAR)
    # __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;

    # get shape of input image, allocate memory for output to which result can be copied to
    shape = imgIn.T.shape
    imgOut = np.empty_like(imgIn)

    # create image buffers which hold images for OpenCL
    imgInBuf = cl.image_from_array(context, ary=imgIn, mode="r", norm_int=True, num_channels=1)
    imgOutBuf = cl.image_from_array(context, ary=imgOut, mode="w", norm_int=True, num_channels=1)

    # load, compile and execute OpenCL program
    program = cl.Program(context, open('kernel.cl').read()).build()
    program.img_rotate(queue, shape, None, imgInBuf, imgOutBuf, np.double(sin_theta), np.double(cos_theta))
    cl.enqueue_copy(queue, imgOut, imgOutBuf, origin=(0, 0), region=shape,
                    is_blocking=True)  # wait until finished copying resulting image back from GPU to CPU

    # write output image
    cv2.imwrite('photographer_rotated.png', imgOut)

    # show images
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(imgIn, cmap='gray')
    ax[1].imshow(imgOut, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
