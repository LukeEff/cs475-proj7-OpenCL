
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <omp.h>

#include "cl.h"
#include "cl_platform.h"


const char *			CL_FILE_NAME = { "fourier.cl" };

// opencl objects:
cl_platform_id		Platform;
cl_device_id		Device;
cl_kernel		Kernel;
cl_program		Program;
cl_context		Context;
cl_command_queue	CmdQueue;


#define F_2_PI			(float)(2.*M_PI)

// files to read and write:
#define BIGSIGNALFILEBIN	(char*)"bigsignal.bin"
#define BIGSIGNALFILEASCII	(char*)"bigsignal.txt"
#define CSVPLOTFILE		(char*)"plot.csv"

// how many elements are in the big signal:
#define NUMELEMENTS	(1*1024*1024)

// only consider this many periods (this is enough to uncover the secret sine waves):
#define MAXPERIODS	100

// which file type to read, BINARY or ASCII (BINARY is much faster to read):
#define BINARY

// TODO: Compare performance metrics for different local sizes
#define LOCAL_SIZE      32

// globals:
float BigSums[MAXPERIODS];		// the overall MAXPERIODS autocorrelation array
float BigSignal[NUMELEMENTS];		// the overall NUMELEMENTS-big signal data

// function prototypes:
void			SelectOpenclDevice();
char *			Vendor( cl_uint );
char *			Type( cl_device_type );
void			Wait( cl_command_queue );


int
main( int argc, char *argv[ ] )
{
#ifndef _OPENMP
    fprintf( stderr, "OpenMP is not enabled!\n" );
    return 1;
#endif

    FILE *fpointer;
    fpointer = fopen(CL_FILE_NAME, "r");
    if( fpointer == NULL )
    {
        fprintf( stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME );
        return 1;
    }

    cl_int status;		// returned status from opencl calls -- test against CL_SUCCESS


    // get the platform id and the device id:
    SelectOpenclDevice();		// sets the global variables Platform and Device

    // Populate the BigSignal array with data from the file
#ifdef ASCII
    FILE *fp = fopen( BIGSIGNALFILEASCII, "r" );
    if( fp == NULL )
    {
        fprintf( stderr, "Cannot open data file '%s'\n", BIGSIGNALFILEASCII );
        return -1;
    }

    for( int i = 0; i < NUMELEMENTS; i++ )
    {
        float f;
        fscanf( fp, "%f", &f );
        BigSignal[i] = f;
    }
#endif
#ifdef BINARY
    FILE *fp = fopen( BIGSIGNALFILEBIN, "rb" );
    if( fp == NULL )
    {
        fprintf( stderr, "Cannot open data file '%s'\n", BIGSIGNALFILEBIN );
        return -1;
    }

    fread( BigSignal, sizeof(float), NUMELEMENTS, fp );
#endif

    // Now that host data is ready, allocate buffers on OpenCL device and write data to the device buffers

    // Create an OpenCL context
    Context = clCreateContext( NULL, 1, &Device, NULL, NULL, &status );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clCreateContext failed (2)\n" );

    // Create a command queue
    CmdQueue = clCreateCommandQueue( Context, Device, 0, &status );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clCreateCommandQueue failed\n" );

    size_t numWorkGroups = NUMELEMENTS / LOCAL_SIZE;
    size_t BigSignalBufferSize = NUMELEMENTS * sizeof(float);
    size_t BigSumsBufferSize = MAXPERIODS * sizeof(float);

    // Allocate device memory
    cl_mem BigSignalBuffer = clCreateBuffer( Context, CL_MEM_READ_ONLY, BigSignalBufferSize, NULL, &status );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clCreateBuffer failed (1)\n" );

    cl_mem BigSumsBuffer = clCreateBuffer( Context, CL_MEM_WRITE_ONLY, BigSumsBufferSize, NULL, &status );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clCreateBuffer failed (2)\n" );

    // Write host data to device buffers

    status = clEnqueueWriteBuffer( CmdQueue, BigSignalBuffer, CL_FALSE, 0, BigSignalBufferSize, BigSignal, 0, NULL, NULL );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clEnqueueWriteBuffer failed (1)\n" );

    Wait( CmdQueue );
    // Read the kernel source file into a string

    fseek( fpointer, 0, SEEK_END );
    size_t fileSize = ftell( fpointer );
    fseek( fpointer, 0, SEEK_SET );
    char *clProgramText = new char [fileSize+1];		// leave room for '\0'
    size_t n = fread( clProgramText, 1, fileSize, fpointer );
    clProgramText[fileSize] = '\0';
    fclose( fpointer );
    if( n != fileSize )
        fprintf( stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n );

    // Create the text for the kernel program

    char *strings[1];
    strings[0] = clProgramText;
    cl_program Program = clCreateProgramWithSource( Context, 1, (const char **)strings, NULL, &status );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clCreateProgramWithSource failed\n" );
    delete [ ] clProgramText;

    // Compile the kernel program

    const char *options = { "" };
    status = clBuildProgram( Program, 1, &Device, options, NULL, NULL );
    if( status != CL_SUCCESS )
    {
        size_t size;
        clGetProgramBuildInfo( Program, Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size );
        cl_char *log = new cl_char [size];
        clGetProgramBuildInfo( Program, Device, CL_PROGRAM_BUILD_LOG, size, log, NULL );
        fprintf( stderr, "clBuildProgram failed:\n%s\n", log );
        delete [ ] log;
    }

    // Create the kernel object

    cl_kernel Kernel = clCreateKernel(Program, "DoLocalFourier", &status);
    if( status != CL_SUCCESS )
        fprintf( stderr, "clCreateKernel failed\n" );

    // Set the kernel arguments

    status = clSetKernelArg( Kernel, 0, sizeof(cl_mem), &BigSignalBuffer );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clSetKernelArg failed (1)\n" );

    // Local memory buffer for each work item to store partial sums, reducing the number of writes to shared global memory
    status = clSetKernelArg( Kernel, 1,  LOCAL_SIZE * MAXPERIODS * sizeof(float), NULL );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clSetKernelArg failed (1)\n" );

    // Sum of all partial sums for each period
    status = clSetKernelArg( Kernel, 2, sizeof(cl_mem), &BigSumsBuffer );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clSetKernelArg failed (2)\n" );

    // Enqueue the kernel object for execution

    size_t globalWorkSize[3] = { NUMELEMENTS, 1, 1 };
    size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

    Wait( CmdQueue );
    double time0 = omp_get_wtime( );

    status = clEnqueueNDRangeKernel( CmdQueue, Kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
    if( status != CL_SUCCESS )
        fprintf( stderr, "clEnqueueNDRangeKernel failed: %d\n", status );

    Wait( CmdQueue );
    printf("F\n");

    // Read the device output buffer to the host output array

    status = clEnqueueReadBuffer( CmdQueue, BigSumsBuffer, CL_TRUE, 0, BigSumsBufferSize, BigSums, 0, NULL, NULL );
    if( status != CL_SUCCESS )
            fprintf( stderr, "clEnqueueReadBuffer failed\n" );

    double time1 = omp_get_wtime( );

	// print the performance:
    double seconds = time1 - time0;
    // Numelements is the number of CUDA threads that ran. Each cuda thread computed MAXPERIODS number of sums.

            double performance = (double)NUMELEMENTS*(double)MAXPERIODS*(double)1./seconds/1000000.;        // mega-mults computed per second
            fprintf( stderr, "%10d elements, %9.2lf mega-multiplies computed per second\n",
         NUMELEMENTS, performance );


	// write the file to be plotted to look for the secret sine wave:
    FILE *fpp = fopen( CSVPLOTFILE, "w" );
    if( fpp == NULL )
    {
        fprintf( stderr, "Cannot write to plot file '%s'\n", CSVPLOTFILE );
    }
    else
    {
        for( int s = 1; s < MAXPERIODS; s++ )		// BigSums[0] is huge -- don't use it
        {
            fprintf( fpp, "%6d , %10.2f\n", s, BigSums[s] );
        }
        fclose( fpp );
    }


	return 0;
}

/////////////////////////////////////////////////////////////////
// Helper functions                                            //
/////////////////////////////////////////////////////////////////

// wait until all queued tasks have taken place:

void
Wait( cl_command_queue queue )
{
    cl_event wait;
    cl_int      status;

    status = clEnqueueMarker( queue, &wait );
    if( status != CL_SUCCESS )
        fprintf( stderr, "Wait: clEnqueueMarker failed\n" );

    status = clWaitForEvents( 1, &wait );
    if( status != CL_SUCCESS )
        fprintf( stderr, "Wait: clWaitForEvents failed\n" );
    if (status == CL_INVALID_VALUE)
        fprintf(stderr, "Wait: clWaitForEvents returned CL_INVALID_VALUE\n");
    if (status == CL_INVALID_CONTEXT)
        fprintf(stderr, "Wait: clWaitForEvents returned CL_INVALID_CONTEXT\n");
    if (status == CL_INVALID_EVENT)
        fprintf(stderr, "Wait: clWaitForEvents returned CL_INVALID_EVENT\n");
    if (status == CL_OUT_OF_RESOURCES)
        fprintf(stderr, "Wait: clWaitForEvents returned CL_OUT_OF_RESOURCES\n");
    if (status == CL_OUT_OF_HOST_MEMORY)
        fprintf(stderr, "Wait: clWaitForEvents returned CL_OUT_OF_HOST_MEMORY\n");
    fprintf(stderr, "Wait: clWaitForEvents returned %d\n", status);
}


// vendor ids:
#define ID_AMD		0x1002
#define ID_INTEL	0x8086
#define ID_NVIDIA	0x10de

void
SelectOpenclDevice()
{
    // select which opencl device to use:
    // priority order:
    //	1. a gpu
    //	2. an nvidia or amd gpu
    //	3. an intel gpu
    //	4. an intel cpu

    int bestPlatform = -1;
    int bestDevice = -1;
    cl_device_type bestDeviceType;
    cl_uint bestDeviceVendor;
    cl_int status;		// returned status from opencl calls
    // test against CL_SUCCESS

    // find out how many platforms are attached here and get their ids:

    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if( status != CL_SUCCESS )
        fprintf(stderr, "clGetPlatformIDs failed (1)\n");

    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if( status != CL_SUCCESS )
        fprintf(stderr, "clGetPlatformIDs failed (2)\n");

    for( int p = 0; p < (int)numPlatforms; p++ )
    {
        // find out how many devices are attached to each platform and get their ids:

        cl_uint numDevices;

        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if( status != CL_SUCCESS )
            fprintf(stderr, "clGetDeviceIDs failed (2)\n");

        cl_device_id* devices = new cl_device_id[numDevices];
        status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
        if( status != CL_SUCCESS )
            fprintf(stderr, "clGetDeviceIDs failed (2)\n");

        for( int d = 0; d < (int)numDevices; d++ )
        {
            cl_device_type type;
            cl_uint vendor;
            size_t sizes[3] = { 0, 0, 0 };

            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);

            clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL);

            // select:

            if( bestPlatform < 0 )		// not yet holding anything -- we'll accept anything
            {
                bestPlatform = p;
                bestDevice = d;
                Platform = platforms[bestPlatform];
                Device = devices[bestDevice];
                bestDeviceType = type;
                bestDeviceVendor = vendor;
            }
            else					// holding something already -- can we do better?
            {
                if( bestDeviceType == CL_DEVICE_TYPE_CPU )		// holding a cpu already -- switch to a gpu if possible
                {
                    if( type == CL_DEVICE_TYPE_GPU )			// found a gpu
                    {										// switch to the gpu we just found
                        bestPlatform = p;
                        bestDevice = d;
                        Platform = platforms[bestPlatform];
                        Device = devices[bestDevice];
                        bestDeviceType = type;
                        bestDeviceVendor = vendor;
                    }
                }
                else										// holding a gpu -- is a better gpu available?
                {
                    if( bestDeviceVendor == ID_INTEL )			// currently holding an intel gpu
                    {										// we are assuming we just found a bigger, badder nvidia or amd gpu
                        bestPlatform = p;
                        bestDevice = d;
                        Platform = platforms[bestPlatform];
                        Device = devices[bestDevice];
                        bestDeviceType = type;
                        bestDeviceVendor = vendor;
                    }
                }
            }
        }
        delete [ ] devices;
    }
    delete [ ] platforms;


    if( bestPlatform < 0 )
    {
        fprintf(stderr, "I found no OpenCL devices!\n");
        exit( 1 );
    }
    else
    {
#ifndef CSV
        fprintf(stderr, "I have selected Platform #%d, Device #%d: ", bestPlatform, bestDevice);
        fprintf(stderr, "Vendor = %s, Type = %s\n", Vendor(bestDeviceVendor), Type(bestDeviceType) );
#endif
    }
}

char *
Vendor( cl_uint v )
{
    switch( v )
    {
        case ID_AMD:
            return (char *)"AMD";
        case ID_INTEL:
            return (char *)"Intel";
        case ID_NVIDIA:
            return (char *)"NVIDIA";
    }
    return (char *)"Unknown";
}

char *
Type( cl_device_type t )
{
    switch( t )
    {
        case CL_DEVICE_TYPE_CPU:
            return (char *)"CL_DEVICE_TYPE_CPU";
        case CL_DEVICE_TYPE_GPU:
            return (char *)"CL_DEVICE_TYPE_GPU";
        case CL_DEVICE_TYPE_ACCELERATOR:
            return (char *)"CL_DEVICE_TYPE_ACCELERATOR";
    }
    return (char *)"Unknown";
}
