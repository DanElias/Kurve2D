extern "C"

#include <math.h>
#include <stdio.h>

__global__ void add(int vertices, int positions_n, int *linear_adjacency_matrix, float *x_positions, float *y_positions, float *x_velocities, float *y_velocities, float *result_positions){
    int index = threadIdx.x + blockIdx.x * blockDim.x; // 0 + 0 * 512
    int id = index;
    float scale = 1;
    float distance_scale = 1;
    float C1 = 2.0 / scale;
    float C2 = 1.0 / scale;
    float C3 = 1.0 / scale;
    float C4 = 0.1 / scale;
    float max_velocity = 100;
    float min_velocity = -100;
    int linear_adjacency_matrix_size = vertices * vertices;
    
    while (id < positions_n && id * vertices + vertices <= linear_adjacency_matrix_size){
        int pos_index;
        int vertex_row = id * vertices; // index where the row starts for the vertex in the adj matrix
        float x = x_positions[id];
        float y = y_positions[id];
        float new_x_velocity = 0;
        float new_y_velocity = 0;

        for(pos_index = 0; pos_index < vertices; pos_index++) {
            //Don't calculate forces with itself
            if(pos_index == id) {
                continue;
            }

            
            //printf("%i", linear_adjacency_matrix[vertex_row + pos_index]);
           
            
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
            if (distance < 14) {
                distance = 14;
            }
            
            // Calculate forces
            float force = 1;
            
            if (isAdj == 1) { 
                // calculate force of attraction
                force = C1 * log10(distance * distance_scale / C2);
            } else {
                // calculate force of repulsion
                force = -50 * C3 / pow(distance / distance_scale, 2);
            }
            
            // Calculate new velocities
            float velocity_x = separation_x * force / distance * C4;
            float velocity_y = separation_y * force / distance * C4;
            
            
            new_x_velocity = new_x_velocity - velocity_x;
            new_y_velocity = new_y_velocity - velocity_y;
            
        }
        
        
        //synchronize all threads, measure speed between sequential and parallel
        __syncthreads();
        
        if (max_velocity > new_x_velocity + x_velocities[id] && new_x_velocity + x_velocities[id] > min_velocity){
            x_velocities[id] = new_x_velocity;
        }

        if (x_positions[id] + x_velocities[id] < 1057.5 && x_positions[id] + x_velocities[id] > 0) {
            x_positions[id] = x_positions[id] + x_velocities[id];
        }
        
        if (max_velocity > new_y_velocity + y_velocities[id] && new_y_velocity + y_velocities[id] > min_velocity){
            y_velocities[id] = new_y_velocity;
        }

        if (y_positions[id] + y_velocities[id] < 675 && y_positions[id] + y_velocities[id] > 0) {
            y_positions[id] = y_positions[id] + y_velocities[id];
        }
        
        
        id =  id + blockDim.x * gridDim.x; // id + 512 * numBlocks    
        
    }
}

