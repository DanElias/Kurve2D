extern "C"

#include <math.h>
#include <stdio.h>

__global__ void add(int vertices, int positions_n, int *linear_adjacency_matrix, float *x_positions, float *y_positions, float *x_velocities, float *y_velocities, float *result_positions){
    int index = threadIdx.x + blockIdx.x * blockDim.x; // 0 + 0 * 512
    int id = index;
    float C1 = 2.0;
    float C2 = 1.0;
    float C3 = 1.0;
    float C4 = 0.1;
    
    while (id < positions_n){
        
        int pos_index;
        int vertex_row = id * vertices; // index where the row starts for the vertex in the adj matrix
        float x = x_positions[id];
        float y = y_positions[id];
        float x_velocity = x_velocities[id];
        float y_velocity = y_velocities[id];
        float new_x_velocity = 0;
        float new_y_velocity = 0;

        for(pos_index = 0; pos_index < vertices; pos_index++) {
            //Don't calculate forces with itself
            if(pos_index == id) {
                continue;
            }
            
            //0 or 1, where 1 is an adjacent vertex in the current vertex row
            int isAdj = linear_adjacency_matrix[vertex_row + pos_index];
            
            // Coordinates of the other vertex
            float other_x = x_positions[pos_index];
            float other_y = y_positions[pos_index];
            
            // Distance between x and y components of two vertices
            float separation_x = x - other_x;
            float separation_y = y - other_y;
            
            // Euclidean Distance
            float distance = sqrt(pow(separation_x,2) + pow(separation_y,2));
            
            // Calculate forces
            float force = 1;
            
            if (isAdj == 1) { 
                // calculate force of attraction
                force = C1 * log10(distance / C2);
            } else {
                // calculate force of repulsion
                force = C3 / pow(distance,2);
            }
            
            // Calculate new velocities
            float velocity_x = separation_x * force / distance * C4;
            float velocity_y = separation_y * force / distance * C4;
            
            // Acum velocities
            
            new_x_velocity = new_x_velocity + velocity_x;
            new_y_velocity = new_y_velocity + velocity_y;
        }
        
        //synchronize all threads, measure speed between sequential and parallel
        __syncthreads();
        
        x_velocities[id] = x_velocities[id] + new_x_velocity;
        y_velocities[id] = y_velocities[id] + new_y_velocity;
        x_positions[id] = x_positions[id] + x_velocities[id];
        y_positions[id] = y_positions[id] + y_velocities[id];

        id =  id + blockDim.x * gridDim.x; // id + 512 * numBlocks    
    }
}

