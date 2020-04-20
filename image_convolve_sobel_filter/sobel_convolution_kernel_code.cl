// * coordinates are pixel-coordinates
// * no interpolation between pixels
// * pixel values from outside of image are taken from edge instead

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;

__constant float soble_filter[3][3] = {{-1, 0, 1},{-2, 0, 2},{-1, 0, 1}};

__kernel void custom_convolution_2d(
    __read_only image2d_t src,
    __write_only image2d_t dst) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);

  int2 coord = (int2){i, j};

  int2 c = (int2){3 / 2, 3 / 2};

  float sum_x = 0;
  float sum_y = 0;
  float sum = 0;
  for (int x = -c.x; x <= c.x; x++) {
    for (int y = -c.y; y <= c.y; y++) {
        int2 kernelCoord = c + (int2)(x,y);
        int2 imageCoord = coord + (int2)(x,y);
        sum_x = sum_x + (soble_filter[kernelCoord.y][kernelCoord.x]
                      * (float)read_imagef(src,sampler,imageCoord).x);
        sum_y = sum_y + (soble_filter[kernelCoord.x][kernelCoord.y] // set filter direction x/y by switching indexing order
                      * (float)read_imagef(src,sampler,imageCoord).x);
    }
  }
  sum = sqrt( pow(sum_x,2) + pow(sum_y,2) );
  write_imagef(dst, coord, (float4)(sum, 0.0f, 0.0f, 0.0f));
}