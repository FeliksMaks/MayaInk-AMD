// maya-ik2 on HIP: the CUDA runtime names Maya's hip_compat/cuda_runtime.h does not map (managed memory, capture mode).
#pragma once
#define cudaMallocManaged hipMallocManaged
#define cudaCpuDeviceId hipCpuDeviceId
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemAdvise hipMemAdvise
#define cudaMemAdviseSetPreferredLocation hipMemAdviseSetPreferredLocation
#define cudaMemAdviseSetAccessedBy hipMemAdviseSetAccessedBy
#define cudaMemAdviseSetReadMostly hipMemAdviseSetReadMostly
#define cudaDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaMemAttachGlobal hipMemAttachGlobal
