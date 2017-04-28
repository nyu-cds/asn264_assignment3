# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import numpy as np
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters


@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    
    '''
    The original mandelbrot function iterated directly over every (x,y) pair in the matrix image. 

    In the CUDA implementation, compute_mandel is a kernel which executes the parallelizable portion of the fractal computation.
    The kernel is executed in parallel (simultaneously) by threads, which are "contained" in grids, containing blocks, 
    which in turn contain threads, on a GPU device. In this version, of mandelbrot computation, each thread is responsible
    for computing certain pixels of the image.

    starting_x and starting_y describe the thread indices of the current thread. 
    Then since image width = 1024, block_width = 32 and grid_width = 32, each thread will be responsible for iterating over
    at most 1024/(32*32)=1 values of real. 
    Since image height = 1536, block_height = 8 and grid_height = 16, each thread will be responsible for iterating over at most 
    1536/(8*16)=12 values of imag. 
    Then each thread will be responsible for at most 12 pixels in the image. By formatting the loops as below, we guarantee 
    that the execution of each thread is independent because each pixel is only computed exactly once over all threads.

    Reference: https://devblogs.nvidia.com/parallelforall/numbapro-high-performance-python-cuda-acceleration/
    '''

    #Set variables for image shape
    height = image.shape[0]
    width = image.shape[1]

    #Compute pixel sizes
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    #get starting x and y
    starting_x, starting_y = cuda.grid(2)

    #values for calculating ending x and y
    block_width = cuda.blockDim.x
    block_height = cuda.blockDim.y
    grid_width = cuda.gridDim.x
    grid_height = cuda.gridDim.y

    for x in range(starting_x, width, grid_width*block_width):
        real = min_x + x * pixel_size_x
        for y in range(starting_y, height, grid_height*block_height):
            imag = min_y + y * pixel_size_y 
            image[y, x] = mandel(real, imag, iters)

    
if __name__ == '__main__':

    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)
    
    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()