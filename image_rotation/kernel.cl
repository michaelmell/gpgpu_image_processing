// (4b) settings of sampler to read pixel values from image: 
// * coordinates are pixel-coordinates
// * no interpolation between pixels
// * pixel values from outside of image are taken from edge instead
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;


__kernel void img_rotate(
    __read_only image2d_t src_data,
    __write_only image2d_t dest_data,
    double sinTheta,
    double cosTheta)
    {
    //Work-item gets its index within index space
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    //Calculate location of data to move into (ix,iy)
    //Output decomposition as mentioned
    float x0 = get_image_width(src_data)/2.0f;
    float y0 = get_image_height(src_data)/2.0f;

    float xOff = ix - x0;
    float yOff = iy - y0;

    int xpos = (int)(xOff*cosTheta + yOff*sinTheta + x0);
    int ypos = (int)(yOff*cosTheta - xOff*sinTheta + y0);

	const float pxVal = read_imagef(src_data, sampler, (int2)(xpos, ypos)).s0;

	write_imagef(dest_data, (int2)(ix, iy), (float4)(pxVal, 0.0f, 0.0f, 0.0f));
}
