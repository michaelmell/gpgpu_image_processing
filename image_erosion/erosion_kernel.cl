// configure sampler to read pixel values from image:
// * coordinates are pixel-coordinates
// * no interpolation between pixels
// * pixel values from outside of image are taken from edge instead
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

// kernel with 2 arguments: input and output image
__kernel void morphOpKernel(__read_only image2d_t in, __write_only image2d_t out)
{
	// IDs of work-item represent x and y coordinates in image
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	// extreme pixel value inside structuring element (1.0f for erosion)
	float extremePxVal = 1.0f;

	// structuring element is square of size 3x3: therefore we simply walk the 3x3 neighborhood of the current location
	for(int i=-1; i<=1; ++i)
	{
		for(int j=-1; j<=1; ++j)
		{
			// get pixel value at location (x+i, y+j)
			const float pxVal = read_imagef(in, sampler, (int2)(x + i, y + j)).s0;

			// keep minimal value in work-item area
            extremePxVal = min(extremePxVal, pxVal);
		}
	}
	
	// write value of pixel to output image at location (x, y)
	write_imagef(out, (int2)(x, y), (float4)(extremePxVal, 0.0f, 0.0f, 0.0f));
}
