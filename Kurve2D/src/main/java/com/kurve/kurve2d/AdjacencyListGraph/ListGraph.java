/* 
 * Graph Adjacency List Class
 * @author: Daniel Elias
 */
package com.kurve.kurve2d.AdjacencyListGraph;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;

import org.json.simple.JSONObject;

import utils.JSONUtils;


/**
 * Adjacency List representation of a Graph
 * @author DanElias
 */
public class ListGraph {
    private HashMap<Integer, ArrayList<Integer>> vertices_mapping; // Matrix mapping
    private HashMap<Integer, Vertex> vertices; // Actual vertex objects
    private HashMap<String, Integer> vertices_ids; // Original id - mat index mapping
    private ArrayList<Edge> edges;
    private int number_of_vertices; // CUDA problem size
    public float[] x_velocities;
    public float[] y_velocities;
    public float[] x_positions;
    public float[] y_positions;
    private List<JSONObject> vertices_list; // Json vertices
    private List<JSONObject> edges_list; // Json edges
    
    public ListGraph(JSONObject graph_json){
        this.vertices_mapping = new HashMap<Integer, ArrayList<Integer>>();
        this.vertices = new HashMap<Integer, Vertex>();
        this.vertices_ids = new HashMap<String, Integer>();
        this.edges = new ArrayList<Edge>();
        
        Object json_vertices = graph_json.get("vertices");
        this.vertices_list = JSONUtils.objectToJSONObjectArrayList(json_vertices);
        
        Object json_edges = graph_json.get("edges");
        this.edges_list = JSONUtils.objectToJSONObjectArrayList(json_edges);
        
        this.number_of_vertices = this.vertices_list.size();
        
        this.x_positions = new float[this.number_of_vertices];
        this.y_positions = new float[this.number_of_vertices];
        this.x_velocities = new float[this.number_of_vertices];
        this.y_velocities = new float[this.number_of_vertices];
        
        setVertices();
        setEdges();
        createPolygonCoordinates((float) 250);
        //printGraph();
        //printXYPositionsMatrices();
    }
    
    /**
     * @author DanElias
     * Parse the json object vertices to transform to java collection
     */
    private void setVertices() {
        Integer index = 0;
        for (JSONObject vertex : this.vertices_list){
            String vertex_id = "";
            if (vertex.get("id") != null) {
                vertex_id = vertex.get("id").toString();
            } else {
                if (vertex.get("name") != null) {
                    vertex_id = vertex.get("name").toString();
                }
            }
            String vertex_value = (vertex.get("value") != null) ? vertex.get("value").toString() : vertex_id;
            Vertex new_vertex = new Vertex(index, vertex_id, vertex_value);
            this.vertices.put(index, new_vertex);
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
            Edge new_edge = new Edge(source, target);
            this.edges.add(new_edge);
            Vertex vertex = this.vertices.get(this.vertices_ids.get(source));
            vertex.addAdjacentVertex(this.vertices_ids.get(target));
        }
    }
    
    /**
     * @author DanElias
     * Prints the graph
     */
    public void printGraph() {
        for (Vertex vertex : this.vertices.values()){
            System.out.println("Id: ");
            System.out.println(vertex.getId());
            System.out.println("Adjacent vertices: ");
            System.out.println(vertex.getAdjacentVertices());
        }
    }
    
    /**
     * @author DanElias
     * Sets the graphs vertices starting positions in a circunference
     */
    private void createPolygonCoordinates(float radius) {
        float sides = this.vertices.size();
        float fpi = (float) Math.PI;
        float angle_per_side = (2 * fpi / sides);
        float initial_angle = fpi / 2;
        //(x,y) = ( r * cos(theta), r * sin(theta))
        for (int i = 0; i < sides; i++){
            float angle = angle_per_side * i + initial_angle;
            float x = (float) (radius * Math.cos(angle));
            float y = (float) (radius * Math.sin(angle));
            this.x_positions[i] = x + 530;
            this.y_positions[i] = y + 350;
        }
    } 
    
    /**
     * @author DanElias
     * Prints the (x,y) coordinates for each vertex
     */
    public void printXYPositionsMatrices() {
        System.out.println("\nX positions: ");
        for (int i = 0; i < this.number_of_vertices; i++){
		for (int j = 0; j < this.number_of_vertices; j++){
			System.out.print("\t" + this.x_positions[i*this.number_of_vertices+j]);
		}
		System.out.println("\n");
	}
        System.out.println("\n");
        System.out.println("Y positions: ");
        for (int i = 0; i < this.number_of_vertices; i++){
		for (int j = 0; j < this.number_of_vertices; j++){
			System.out.print("\t" + this.y_positions[i*this.number_of_vertices+j]);
		}
		System.out.println("\n");
	}
        System.out.println("\n");
    }
    
    /**
     * @author
     * @return number of vertices in the graph 
     */
    public int getNumberOfVertices(){
        return this.number_of_vertices;
    }
    
    /**
     * @author DanElias
     * @return reference to the x positions array
     */
    public float[] getXPositions() {
        return this.x_positions;
    }
    
    /**
     * @author DanElias
     * @return the reference to the y positions array
     */
    public float[] getYPositions() {
        return this.y_positions;
    }
    
    /**
     * @author DanElias
     * @return the reference to the x velocities array
     */
    public float[] getXVelocities() {
        return this.x_velocities;
    }
    
    /**
     * @author DanElias
     * @return the reference to the y velocities array
     */
    public float[] getYVelocities() {
        return this.y_velocities;
    }
}
