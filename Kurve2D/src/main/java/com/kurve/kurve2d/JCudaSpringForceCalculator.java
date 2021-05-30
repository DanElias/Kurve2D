/* 
 * Force Directed Graph Parallel Calculator in the GPU
 * @author: Daniel Elias
 */
package com.kurve.kurve2d;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.IOException;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import utils.JCudaSamplesUtils;

import com.kurve.kurve2d.Graph.MatrixGraph;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;

/**
 * @author DanElias
 */
public class JCudaSpringForceCalculator {
    private String ptxFileName;
    private CUdevice device;
    private CUcontext context;
    private CUmodule module;
    private CUfunction function;
    private int N;
    private int positions_n;
    private int size_bytes;
    private int size_bytes_linear_adjacency_matrix;
    private float[] x_positions;
    private float[] y_positions;
    private float[] x_velocities;
    private float[] y_velocities;
    private int[] linear_adjacency_matrix;
    private final int THREADS_PER_BLOCK = 512; //threads per block blockSizeX 256
    private int GRID_SIZE; //gridSizeX
    
    private CUdeviceptr device_linear_adjacency_matrix;
    private CUdeviceptr device_x_positions;
    private CUdeviceptr device_y_positions;
    private CUdeviceptr device_x_velocities;
    private CUdeviceptr device_y_velocities;
    private CUdeviceptr device_result_positions;
    float result_positions[];
    
    /**
     * Entry point
     * @param ptxFileName compiled ptx filename if already exists
     * @param N size of the problem
     * @throws IOException If an IO error occurs
     */
    public JCudaSpringForceCalculator(
            String ptxFileName, // ptx filename url
            int N, // num of vertices * num of vertices
            int positions_n, // n * n = size of x/y positions matrix
            int[] linear_adjacency_matrix, // adjacency matrix graph
            float[] x_positions,
            float[] y_positions,
            float[] x_velocities,
            float[] y_velocities) throws IOException {
        
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        
        if(ptxFileName.isEmpty()) {
            // Create the PTX file by calling the NVCC
            this.ptxFileName = JCudaSamplesUtils.preparePtxFile(
            "src/main/java/com/kurve/kurve2d/CudaKernels/JCudaSpringForceCalculatorKernel.cu");
            // System.out.println(ptxFileName);
            // src/main/java/com/kurve/kurve2d/CudaKernels/JCudaSpringForceCalculatorKernel.ptx
        } else {
            this.ptxFileName = ptxFileName;
        }

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        this.device = new CUdevice();
        cuDeviceGet(this.device, 0);
        this.context = new CUcontext();
        cuCtxCreate(this.context, 0, this.device);
        
        // Load the ptx file.
        this.module = new CUmodule();
        cuModuleLoad(this.module, this.ptxFileName);
        
        // Obtain a function pointer to the "add" function.
        this.function = new CUfunction();
        cuModuleGetFunction(this.function, this.module, "add");
        
        // Initialize problem size N
        this.N = N;
        this.positions_n = positions_n;
        this.size_bytes = this.positions_n * Sizeof.FLOAT;
        this.size_bytes_linear_adjacency_matrix = this.N * this.N * Sizeof.INT;
        this.linear_adjacency_matrix = linear_adjacency_matrix;
        this.x_positions = x_positions;
        this.y_positions = y_positions;
        this.x_velocities = x_velocities;
        this.y_velocities = y_velocities;
        
        this.GRID_SIZE = (int) Math.ceil((float) (((float)this.positions_n) / this.THREADS_PER_BLOCK)); //gridSizeX
        
        System.out.println(this.GRID_SIZE);
        
        //*** Host variables **//
        this.result_positions = new float[this.positions_n];
        
        //*** Device variables **//
        // Allocate Device Linear Adjacency Matrix
        this.device_linear_adjacency_matrix = new CUdeviceptr();
        cuMemAlloc(this.device_linear_adjacency_matrix, this.size_bytes_linear_adjacency_matrix);
        
        // Allocate Device X positions
        this.device_x_positions = new CUdeviceptr();// ptr to device variable
        cuMemAlloc(this.device_x_positions, this.size_bytes); // cudaMalloc
        
        // Allocate Device Y positions
        this.device_y_positions = new CUdeviceptr(); 
        cuMemAlloc(this.device_y_positions, this.size_bytes);
        
        // Allocate Device X velocities
        this.device_x_velocities = new CUdeviceptr();
        cuMemAlloc(this.device_x_velocities, this.size_bytes);
        
        // Allocate Device Y velocities
        this.device_y_velocities = new CUdeviceptr(); 
        cuMemAlloc(this.device_y_velocities, this.size_bytes);
        
        // Allocate device output memory
        this.device_result_positions = new CUdeviceptr();
        cuMemAlloc(this.device_result_positions, this.size_bytes);
        
        
        // DEBUG
        System.out.println("*Matrix graph num of vertices: " + this.N);
        System.out.println("*n* n size of x y positions matrix: " + this.positions_n);
        System.out.println("Adj Matrix rows " + this.linear_adjacency_matrix.length);
        System.out.println("X pos size " + this.x_positions.length);
        System.out.println("Y pos size " + this.y_positions.length);
        System.out.println("X vel size " + this.x_velocities.length);
        System.out.println("Y vel size " + this.y_velocities.length);
        
    }
    
    public void calculate(){
        cuCtxSetCurrent(this.context);
        
        /*
        printXYPositionsMatrices(this.x_positions,"X positions before");
        printXYPositionsMatrices(this.y_positions,"Y positions before");
        printXYPositionsMatrices(this.x_velocities,"X velocities before");
        printXYPositionsMatrices(this.y_velocities,"Y velocities before");
        */

        //*** Cuda Memcpy Host to Device **//
        // Copy the host input data to the device variables
        cuMemcpyHtoD( //cudaMemcpy HostToDevice
            this.device_linear_adjacency_matrix, 
            Pointer.to(this.linear_adjacency_matrix), 
            this.size_bytes_linear_adjacency_matrix);
        
        cuMemcpyHtoD( //cudaMemcpy HostToDevice
            this.device_x_positions, 
            Pointer.to(this.x_positions), 
            this.size_bytes);
       
        cuMemcpyHtoD( //cudaMemcpy HostToDevice
            this.device_y_positions,
            Pointer.to(this.y_positions),
            this.size_bytes);
        
        cuMemcpyHtoD( //cudaMemcpy HostToDevice
            this.device_x_velocities, 
            Pointer.to(this.x_velocities), 
            this.size_bytes);
       
        cuMemcpyHtoD( //cudaMemcpy HostToDevice
            this.device_y_velocities,
            Pointer.to(this.y_velocities),
            this.size_bytes);
        
        //*** Kernel Params **//
        Pointer kernelParameters = Pointer.to(
        // params to be sent to kernel __global_
            Pointer.to(new int[]{this.N}), // number of vertices
            Pointer.to(new int[]{this.positions_n}), // size n * n of positions matrices
            Pointer.to(this.device_linear_adjacency_matrix),
            Pointer.to(this.device_x_positions),
            Pointer.to(this.device_y_positions),
            Pointer.to(this.device_x_velocities),
            Pointer.to(this.device_y_velocities),
            Pointer.to(this.device_result_positions)
        );
        
        //*** Kernel Call **// 
        // Call the kernel function.
        cuLaunchKernel(this.function,
            GRID_SIZE,  1, 1,      // Grid dimension
            THREADS_PER_BLOCK, 1, 1,// Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        
        // Synchronization
        cuCtxSynchronize();

        // Copy the device output to the host.
        cuMemcpyDtoH(
            Pointer.to(this.result_positions),
            this.device_result_positions,
            this.size_bytes);
        
        // Copy the device output to the host.
        cuMemcpyDtoH(
            Pointer.to(this.x_positions),
            this.device_x_positions,
            this.size_bytes);
        
        // Copy the device output to the host.
        cuMemcpyDtoH(
            Pointer.to(this.y_positions),
            this.device_y_positions,
            this.size_bytes);
        
        // Copy the device output to the host.
        cuMemcpyDtoH(
            Pointer.to(this.x_velocities),
            this.device_x_velocities,
            this.size_bytes);
        
        // Copy the device output to the host.
        cuMemcpyDtoH(
            Pointer.to(this.y_velocities),
            this.device_y_velocities,
            this.size_bytes);
        
        
        // Verify the result
        /*
        printXYPositionsMatrices(this.x_positions,"X positions after");
        printXYPositionsMatrices(this.y_positions,"Y positions after");
        printXYPositionsMatrices(this.x_velocities,"X velocities after");
        printXYPositionsMatrices(this.y_velocities,"Y velocities after");
        */
    }
    
    public void free() {
        // Clean up.
        cuMemFree(this.device_linear_adjacency_matrix);
        cuMemFree(this.device_x_positions);
        cuMemFree(this.device_y_positions);
        cuMemFree(this.device_result_positions);
    }
    
    public void printXYPositionsMatrices(float[] results, String text) {
        int n = (int) Math.sqrt(this.positions_n);
        System.out.println("\n results " + text + ": ");
        for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			System.out.print("\t" + results[i*n+j]);
		}
		System.out.println("\n");
	}
        System.out.println("\n");
    }
}

