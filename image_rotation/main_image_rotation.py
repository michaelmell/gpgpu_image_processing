import cv2
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

"""
Here we could use a variable sampler:
    self.sampler = cl.Sampler(self.ctx,True,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)
"""

def main():
    # prepare data
    imgIn = cv2.imread('photographer.png', cv2.IMREAD_GRAYSCALE)
    imgIn = imgIn[128:512, :]

    rotation_angle = np.pi/4
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    # setup OpenCL
    platforms = cl.get_platforms()  # a platform corresponds to a driver (e.g. AMD, NVidia, Intel)
    platform = platforms[0]  # take first platform
    devices = platform.get_devices(cl.device_type.GPU)  # get GPU devices of selected platform
    device = devices[0]  # take first GPU
    context = cl.Context([device])  # put selected GPU into context object
    queue = cl.CommandQueue(context, device)  # create command queue for selected GPU and context

    # get shape of input image, allocate memory for output to which result can be copied to
    shape = imgIn.T.shape
    imgOut = np.empty_like(imgIn)

    # (2) create image buffers which hold images for OpenCL
    imgInBuf = cl.image_from_array(context, ary=imgIn, mode="r", norm_int=True, num_channels=1)
    imgOutBuf = cl.Image(context, cl.mem_flags.WRITE_ONLY,
                         cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8),
                         shape=shape)  # placeholder for gray-valued image of given shape

    # (3) load and compile OpenCL program
    program = cl.Program(context, open('kernel.cl').read()).build()

    # (3) from OpenCL program, get kernel object and set arguments (input image, operation type, output image)
    kernel = cl.Kernel(program, 'img_rotate')  # name of function according to kernel.py
    kernel.set_arg(0, imgInBuf)  # input image buffer
    kernel.set_arg(1, imgOutBuf)  # output image buffer
    kernel.set_arg(2, np.double(sin_theta))  # sinTheta
    kernel.set_arg(3, np.double(cos_theta))  # cosTheta


    # (4) copy image to device, execute kernel, copy data back
    # cl.enqueue_copy(queue, imgInBuf, imgIn, origin=(0, 0), region=shape,
    #                 is_blocking=False)  # copy image from CPU to GPU
    cl.enqueue_nd_range_kernel(queue, kernel, shape,
                               None)  # execute kernel, work is distributed across shape[0]*shape[1] work-items (one work-item per pixel of the image)
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
