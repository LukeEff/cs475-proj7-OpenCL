#define IN
#define OUT

#define MAXPERIODS 100
#define F_2_PI			(float)(2.*M_PI)

void 
atomic_add_f(volatile global float* addr, const float val) {
    union {
        uint  u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        next.f32 = (expected.f32=current.f32)+val; // ...*val for atomic_mul_f()
        current.u32 = atomic_cmpxchg((volatile global uint*)addr, expected.u32, next.u32);
    } while(current.u32!=expected.u32);
}

kernel
void
DoLocalFourier(IN global const float *signal, local float *prods, OUT global float *sums)
{
    int gid = get_global_id( 0 );
    int numItems = get_local_size( 0 );
    int tnum = get_local_id( 0 );
    int wgNum = get_group_id( 0 );

    for( int p = 1; p < MAXPERIODS; p++ )
    {
        float omega = F_2_PI/(float)p;
        float time = (float) gid;
        float sum = signal[gid] * sin( omega*time );

        // Slowest solution: atomic add to global memory for each GPU thread
        //atomic_add_f(&sums[p], sum);

        int threadOffset = tnum * MAXPERIODS;
        prods[p + threadOffset] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for( int offset = 1; offset < numItems; offset *= 2)
    {
        int mask = 2*offset - 1;
        barrier(CLK_LOCAL_MEM_FENCE); // WAIT FOR ALL THREADS TO GET HERE

        if( ( tnum & mask ) == 0)
        {
            for( int p = 1; p < MAXPERIODS; p++ )
            {
                prods[p + (tnum * MAXPERIODS)] += prods[p + ((tnum + offset) * MAXPERIODS)];
            }
        } 
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(tnum == 0)
    {
        for( int p = 1; p < MAXPERIODS; p++ )
        {
            // One atomic add per period element per workgroup
            atomic_add_f(&sums[p], prods[p]);
        }
    }
}