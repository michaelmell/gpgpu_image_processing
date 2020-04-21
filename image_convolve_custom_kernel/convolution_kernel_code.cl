// configure sampler to read pixel values from image:
// * coordinates are pixel-coordinates
// * no interpolation between pixels
// * pixel values from outside of image are taken from edge instead
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;

// convolution kernel with 3 arguments: source-image, kernel-image, output-image
__kernel void custom_convolution_2d(
    __read_only image2d_t src,
    __read_only image2d_t kernelImage,
    __write_only image2d_t dst
) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  int2 coord = (int2){i, j}; // work-item coordinate

  const int kernelImageWidth = get_image_width(kernelImage);
  const int kernelImageHeight = get_image_height(kernelImage);

  int2 c = (int2){kernelImageWidth / 2, kernelImageHeight / 2}; // center-coordinate of kernel-image

  float sum = 0;
  for (int x = -c.x; x <= c.x; x++) {
    for (int y = -c.y; y <= c.y; y++) {
        int2 kernelCoord = c + (int2)(x,y);
        int2 imageCoord = coord + (int2)(x,y);
        sum = sum + ((float)read_imagef(kernelImage,sampler,kernelCoord).x
                  * (float)read_imagef(src,sampler,imageCoord).x);
    }
  }
  write_imagef(dst, coord, (float4)(sum, 0.0f, 0.0f, 0.0f));
}