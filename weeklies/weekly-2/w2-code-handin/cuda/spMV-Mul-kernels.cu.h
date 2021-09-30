#ifndef SP_MV_MUL_KERS
#define SP_MV_MUL_KERS

__global__ void
replicate0(int tot_size, char* flags_d) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    if (gid < tot_size) {
        flags_d[gid] = 0;
    }
}

__global__ void
mkFlags(int mat_rows, int* mat_shp_sc_d, char* flags_d) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    if (gid < mat_rows) {
        flags_d[mat_shp_sc_d[gid]] = 1;
    }
}

__global__ void 
mult_pairs(int* mat_inds, float* mat_vals, float* vct, int tot_size, float* tmp_pairs) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    // let prods = map (\(i,x) -> x*vct[i]) mat_val
    if (gid < tot_size) {
        tmp_pairs[gid] = mat_vals[gid] * vct[ mat_inds[gid] ];
    }
}

__global__ void
select_last_in_sgm(int mat_rows, int* mat_shp_sc_d, float* tmp_scan, float* res_vct_d) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    if (gid < mat_rows) { 
        const unsigned int id = mat_shp_sc_d[gid] - 1; 
        res_vct_d[gid] = tmp_scan[id];
    }
}

#endif
