// * coordinates are pixel-coordinates
// * no interpolation between pixels
// * pixel values from outside of image are taken from edge instead
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_LINEAR | CLK_ADDRESS_REPEAT;

/**
 * Rotate image using normalized coordinates for the sampler.
 * Normalized coordinates are needed to be able to use repeating
 * boundaries (i.e. for using CLK_ADDRESS_REPEAT).
 */
__kernel void img_rotate(
    __read_only image2d_t src_data,
    __write_only image2d_t dest_data,
    double sinTheta,
    double cosTheta)
    {
        //Work-item gets its index within index space
        const int ix = get_global_id(0);
        const int iy = get_global_id(1);

        //Calculate location of data to move into (ix,iy) using normalized coordinates
        float imageWidth = get_image_width(src_data);
        float imageHeight = get_image_height(src_data);

        float xCenter = .5f;
        float yCenter = .5f;

        float xOffset = ix/imageWidth - xCenter;
        float yOffset = iy/imageHeight - yCenter;

        float xpos = (xOffset*cosTheta + yOffset*sinTheta + xCenter);
        float ypos = (yOffset*cosTheta - xOffset*sinTheta + yCenter);

        const float pxVal = read_imagef(src_data, sampler, (float2)(xpos, ypos)).s0;

        write_imagef(dest_data, (int2)(ix, iy), (float4)(pxVal, 0.0f, 0.0f, 0.0f));
}
