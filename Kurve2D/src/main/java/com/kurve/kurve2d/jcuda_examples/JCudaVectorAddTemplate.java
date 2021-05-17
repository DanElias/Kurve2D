/*
 *
 */
package com.kurve.kurve2d.jcuda_examples;

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

/**
 */
public class JCudaVectorAddTemplate {
    private String ptxFileName;
    private CUdevice device;
    private CUcontext context;
    private CUmodule module;
    private CUfunction function;
    private int N;
    private int size_bytes;
    
    /**
     * Entry point
     * @param ptxFileName compiled ptx filename if already exists
     * @param N size of the problem
     * @throws IOException If an IO error occurs
     */
    public JCudaVectorAddTemplate(String ptxFileName, int N) throws IOException {
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
        this.size_bytes = this.N * Sizeof.FLOAT;
    }
    
    public void calculate(){
        
        //*** Host variables **//
        // Allocate host input data
        float hostInputA[] = new float[this.N];
        float hostInputB[] = new float[N];
        // Allocate host output memory
        float hostOutput[] = new float[this.N];
        
        //*** Device variables **//
        // Allocate device input data
        CUdeviceptr deviceInputA = new CUdeviceptr();
        cuMemAlloc(deviceInputA, this.size_bytes); // cudaMalloc
        CUdeviceptr deviceInputB = new CUdeviceptr(); // ptr to device variable
        cuMemAlloc(deviceInputB, this.size_bytes); // cudaMalloc
        // Allocate device output memory
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, this.size_bytes);

        
        // Fill host input data
        for(int i = 0; i < this.N; i++){
            hostInputA[i] = (float)i;
            hostInputB[i] = (float)i;
        }

        // Copy the host input data to the device variables
        cuMemcpyHtoD( //cudaMemcpy HostToDevice
            deviceInputA, 
            Pointer.to(hostInputA), 
            this.N * Sizeof.FLOAT);
       
        cuMemcpyHtoD( //cudaMemcpy HostToDevice
            deviceInputB,
            Pointer.to(hostInputB),
            this.size_bytes);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new int[]{this.N}), // device problem size
            Pointer.to(deviceInputA), // params to be sent to kernel __global__
            Pointer.to(deviceInputB),
            Pointer.to(deviceOutput)
        );

        // Blocks and Grid sizes
        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)this.N / blockSizeX);
        
        // Call the kernel function.
        cuLaunchKernel(this.function,
            gridSizeX,  1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, null,               // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        
        // Synchronization
        cuCtxSynchronize();

        // Copy the device output to the host.
        cuMemcpyDtoH(
            Pointer.to(hostOutput),
            deviceOutput,
            this.size_bytes);

        
        // Verify the result
        boolean passed = true;
        for(int i = 0; i < this.N; i++)
        {
            float expected = i+i;
            if (Math.abs(hostOutput[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index "+i+ " found "+hostOutput[i]+
                    " but expected "+expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        // Clean up.
        cuMemFree(deviceInputA);
        cuMemFree(deviceInputB);
        cuMemFree(deviceOutput);
    }
}

