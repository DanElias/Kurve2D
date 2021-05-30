/* 
 * Force Directed Graph Iterative Calculator in the CPU
 * @author: Daniel Elias
 */
package com.kurve.kurve2d;

/**
 * Force Directed Graph Iterative Calculator in the CPU N^2 BIG O
 * @author DanElias
 */
public class IterativeSpringForceCalculator {
    private int N; // num of vertices
    private float[] x_positions;
    private float[] y_positions;
    private float[] x_velocities;
    private float[] y_velocities;
    private int[] linear_adjacency_matrix; // adjacency matrix graph
    
    public IterativeSpringForceCalculator(
            int N, // num of vertices
            int[] linear_adjacency_matrix, // adjacency matrix graph
            float[] x_positions,
            float[] y_positions,
            float[] x_velocities,
            float[] y_velocities) {
        this.N = N;
        this.linear_adjacency_matrix = linear_adjacency_matrix;
        this.x_positions = x_positions;
        this.y_positions = y_positions;
        this.x_velocities = x_velocities;
        this.y_velocities = y_velocities;
    }
    
    public void calculate(
            int vertices,
            int[] linear_adjacency_matrix,
            float[] x_positions,
            float[] y_positions,
            float[] x_velocities,
            float[] y_velocities) {
        float scale = 1;
        float distance_scale = 1;
        float C1 = (float) 2.0 / scale;
        float C2 = (float) 1.0 / scale;
        float C3 = (float) 1.0 / scale;
        float C4 = (float) 0.1 / scale;
        float max_velocity = 100;
        float min_velocity = -100;
        
        for(int id = 0; id < vertices; id++) {
            int vertex_row = id * vertices; // index where the row starts for the vertex in the adj matrix
            float x = x_positions[id]; // current vertex x positions
            float y = y_positions[id]; // current vertex y position
            float new_x_velocity = 0; // the new velocity the x component will have
            float new_y_velocity = 0; // the new velocity the y component will have

            for(int pos_index = 0; pos_index < vertices; pos_index++) {
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
                float distance = (float) Math.sqrt(Math.pow( separation_x,2) + Math.pow(separation_y,2));
                if (distance < 14) {
                    distance = 14;
                }

                // Calculate forces
                float force = 1;

                if (isAdj == 1) { 
                    // calculate force of attraction
                    force = (float) (C1 * Math.log10(distance * distance_scale / C2));
                } else {
                    // calculate force of repulsion
                    force = -50 *  (float) (C3 / Math.pow(distance / distance_scale, 2));
                }

                // Calculate new velocities
                float velocity_x = separation_x * force / distance * C4;
                float velocity_y = separation_y * force / distance * C4;

                new_x_velocity = new_x_velocity - velocity_x;
                new_y_velocity = new_y_velocity - velocity_y;
            }

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
        }
    }
}
