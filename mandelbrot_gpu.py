# 
# A CUDA version to calculate the Mandelbrot set
#
# Aditi Nair
# Assignment 12
# April 30 2017
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
    In the CUDA implementation, compute_mandel is a kernel which executes the parallelizable portion of the fractal computation.
    The kernel is executed in parallel (simultaneously) by threads, which are "located" within a hierarchy of grids and blocks.
    In this version of mandelbrot computation, each thread is responsible for computing certain pixels of the image.

    Since griddim is (32,16) we know the grid is split into 32x16 blocks for a total of 512 blocks.
    Since blockdim is (32,8) we know each block contains 32x8 threads for a total of 256 threads per block. 
    Finally, there are 512*256 = 131072 threads, which must be responsible for computing 1024*1536=1572864 pixels.

    starting_x and starting_y describes the absolute position of the current thread in the grid.
    By the grid arrangement of the threads, we know that starting_x is in the range [0,1024) (since 32x32=1024).
    Likewise, we know that starting_y is in the range [0,128) (since 16*8=128).
    Then, in the image matrix, there are 1536 elements in each row. 
    So we can design each thread to compute up to ceiling(1536/1024) = 2 values along the rows of the image. 
    In the image matrix, there are 1024 elements in each column. 
    So we can design each thread can compute up to ceiling(1024/128) = 8 values along the columns of the image. 
    Then each pixel will be computed exactly once by some thread.

    This can be done by iterating through values of x in range(starting_x, 1536, 1024)
    and by iterating through values of y in range(starting_y, 1024, 128) and computing
    the pixel value at image[y,x] using the mandel function, which is exactly what happens below.

    References: 
    https://devblogs.nvidia.com/parallelforall/numbapro-high-performance-python-cuda-acceleration/
    https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
    '''

    #Set variables for image shape
    height = image.shape[0]
    width = image.shape[1]

    #Compute pixel sizes
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    #Get starting x and y
    starting_x, starting_y = cuda.grid(2)

    #Get grid and block dimensions
    block_width = cuda.blockDim.x
    block_height = cuda.blockDim.y
    grid_width = cuda.gridDim.x
    grid_height = cuda.gridDim.y

    #Loop through pixels to compute mandel for, as described above
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