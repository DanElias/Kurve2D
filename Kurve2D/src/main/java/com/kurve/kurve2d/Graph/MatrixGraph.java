/* 
 * Graph Adjancency Matrix Class
 * @author: Daniel Elias
 */
package com.kurve.kurve2d.Graph;

import java.util.HashMap;
import java.util.List;
import org.json.simple.JSONObject;
import utils.JSONUtils;

/**
 * Graph Adjancency Matrix Class
 * @author DanElias
 */
public class MatrixGraph{
    private HashMap<String, Integer> vertices_ids; // Original id - mat index mapping
    // The adjancency matrix is represented in its two ways as a matrix 
    // or as a one dimensional linearized matrix to be used by CUDA 
    private int[][] adjacency_matrix;
    public int[] linear_adjacency_matrix;
    private int N; // CUDA problem size, adjacency matrix size
    private int number_of_vertices; // number of vertices
    private List<JSONObject> vertices_list; // Json vertices
    private List<JSONObject> edges_list; // Json edges
    
    public MatrixGraph(JSONObject graph_json){
        this.vertices_ids = new HashMap<>();
        
        Object json_vertices = graph_json.get("vertices");
        this.vertices_list = JSONUtils.objectToJSONObjectArrayList(json_vertices);
        
        Object json_edges = graph_json.get("edges");
        this.edges_list = JSONUtils.objectToJSONObjectArrayList(json_edges);
        
        this.number_of_vertices = this.vertices_list.size();
        this.N = this.number_of_vertices * this.number_of_vertices;
        
        this.adjacency_matrix = new int[this.number_of_vertices][this.number_of_vertices];
        this.linear_adjacency_matrix = new int[this.N];
        
        setVertices();
        setEdges();
        //printAdjacencyMatrix();
        //printLinearAdjacencyMatrix();
    }
    
    /**
     * @author DanElias
     * Parse the json object vertices to transform to java collection
     */
    private void setVertices() {
        int index = 0;
        for (JSONObject vertex : this.vertices_list){
            String vertex_id = "";
            if (vertex.get("id") != null) {
                vertex_id = vertex.get("id").toString();
            } else {
                if (vertex.get("name") != null) {
                    vertex_id = vertex.get("name").toString();
                }
            }
            this.vertices_ids.put(vertex_id, index);
            index++;
        }
    }
    
    /**
     * @author DanElias
     * Parse the json object edges to transform to java collection
     */
    private void setEdges() {
        for (JSONObject edge : this.edges_list){
            String source = edge.get("source").toString();
            String target = edge.get("target").toString();
            int source_vertex_id = this.vertices_ids.get(source);
            int target_vertex_id = this.vertices_ids.get(target);
            // Undirected graph, add both ways
            addEdge(source_vertex_id, target_vertex_id);
            addEdge(target_vertex_id, source_vertex_id);
        }
    }
    
    /**
     * @author DanElias
     * Adds edge to the adjancency matrix as a 1
     * Undirected Graph representation
     */
    public void addEdge(int source, int target) {
        this.linear_adjacency_matrix[source * this.number_of_vertices+target] = 1; // linear matrix
        adjacency_matrix[source][target] = 1;
        adjacency_matrix[source][target] = 1;
    }
    
    /**
     * @author DanElias
     * Prints the graph using its adjacency matrix
     */
    private void printAdjacencyMatrix() {
        System.out.println("\nAdjacency matrix: ");
        for (int i = 0; i < this.number_of_vertices; i++){
		for (int j = 0; j < this.number_of_vertices; j++){
			System.out.print("\t" + this.adjacency_matrix[i][j]);
		}
		System.out.println("\n");
	}
        System.out.println("\n");
    }
    
    /**
     * @author DanElias
     * Prints the graph using its linear adjacency matrix
     */
    private void printLinearAdjacencyMatrix() {
        System.out.println("\nLinear adjacency matrix: ");
        for (int i = 0; i < this.number_of_vertices; i++){
		for (int j = 0; j < this.number_of_vertices; j++){
			System.out.print("\t" + this.linear_adjacency_matrix[i*this.number_of_vertices+j]);
		}
		System.out.println("\n");
	}
        System.out.println("\n");
    }
    
    /**
     * @author DanElias
     * @return the size of the matrix = n vertices * n vertices 
     */
    public int getN(){
        return this.N;
    }
    
    /**
     * @author DanElias
     * @return number of vertices = width of matrix
     */
    public int getNumberOfVertices(){
        return this.number_of_vertices;
    }
    
    /**
     * @author DanElias
     * @return number of edges
     */
    public int getNumberOfEdges(){
        return this.edges_list.size();
    }
    
    /**
     * @author DanElias
     * @return reference to the adjancency matrix
     */
    public int[][] getAdjacencyMatrix() {
        return this.adjacency_matrix;
    }
    
    /**
     * @author DanElias
     * @return reference to the linearized adjancency matrix
     */
    public int[] getLinearAdjacencyMatrix() {
        return this.linear_adjacency_matrix;
    }
}
