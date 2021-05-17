extern "C"

__global__ void add(int vertices, int positions_n, int *linear_adjacency_matrix, float *x_positions, float *y_positions, float *result_positions){
    int index = threadIdx.x + blockIdx.x * blockDim.x; // 0 + 0 * 512
    int id = index;
    while (id < positions_n){
        result_positions[id] = x_positions[id] + y_positions[id];
        result_positions[id] = linear_adjacency_matrix[threadIdx.x * vertices];
        id =  id + blockDim.x * gridDim.x; // id + 512 * numBlocks    
    }
}
