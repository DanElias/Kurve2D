/* 
 * Graph Adjacency List Class
 * @author: Daniel Elias
 */
package com.kurve.kurve2d.AdjacencyListGraph;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;

import org.json.simple.JSONObject;

import utils.Utils;

import java.lang.Math;
import java.util.HashSet;
import utils.JSONUtils;

import com.kurve.kurve2d.AdjacencyListGraph.Vertex;
import com.kurve.kurve2d.AdjacencyListGraph.Edge;

/**
 *
 * @author DanElias
 */
public class ListGraph {
    private HashMap<Integer, ArrayList<Integer>> vertices_mapping; // Matrix mapping
    private HashMap<Integer, Vertex> vertices; // Actual vertex objects
    private HashMap<String, Integer> vertices_ids; // Original id - mat index mapping
    private ArrayList<Edge> edges;
    private int N; // CUDA problem size
    private int n; // number of vertices
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
        
        this.N = calculateN();
        
        this.x_positions = new float[this.N];
        this.y_positions = new float[this.N];
        this.x_velocities = new float[this.N];
        this.y_velocities = new float[this.N];
        
        setVertices();
        setEdges();
        setNonAdjacentLists();
        createPolygonCoordinates((float) 250);
        //setXYInitialPositionsMatrices();
        //setXYInitialVelocitiesMatrices();
        //printGraph();
        // printXYPositionsMatrices();
    }
    
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
    
    public void setNonAdjacentLists() {
        for (Vertex vertex : this.vertices.values()){
            ArrayList<Integer> non_adjacent = 
                    new ArrayList<Integer>(this.vertices_ids.values());
            ArrayList<Integer> adjacents_and_self =
                    new ArrayList(vertex.getAdjacentVertices());
            adjacents_and_self.add(vertex.getId());
            non_adjacent.removeAll(vertex.getAdjacentVertices());
            vertex.setNonadjacentVertices(non_adjacent);
        }
    }
    
    public void printGraph() {
        for (Vertex vertex : this.vertices.values()){
            System.out.println("Id: ");
            System.out.println(vertex.getId());
            System.out.println("Adjacent vertices: ");
            System.out.println(vertex.getAdjacentVertices());
            System.out.println("Nonadjacent vertices: ");
            System.out.println(vertex.getNonadjacentVertices());
        }
    }
    
    public void createPolygonCoordinates(float radius) {
        float sides = this.vertices.size();
        float fpi = (float) Math.PI;
        float angle_per_side = (2 * fpi / sides);
        float initial_angle = fpi / 2;
        //(x,y,z) = ( r * cos(theta), r * sin(theta), 0)
        for (int i = 0; i < sides; i++){
            float angle = angle_per_side * i + initial_angle;
            float x = (float) (Math.cos(angle) * radius);
            float y = (float) (Math.sin(angle) * radius);
            System.out.println();
            this.x_positions[i] = x + 530;
            this.y_positions[i] = y + 350;
        }
    } 
    
    public void setXYInitialPositionsMatrices() {
        for (int i = 0; i < this.N; i++){
            this.x_positions[i] = (float) i + i * 94;
	}
        
        for (int i = 0; i < this.N; i++){
            this.y_positions[i] = (float) this.N - i + i * 36;
	}
    }
    
    public void setXYInitialVelocitiesMatrices() {
        for (int i = 0; i < this.N; i++){
            this.x_velocities[i] = 0;
	}
        
        for (int i = 0; i < this.N; i++){
            this.y_velocities[i] = 0;
	}
    }
    
    public void printXYPositionsMatrices() {
        System.out.println("\nX positions: ");
        for (int i = 0; i < this.n; i++){
		for (int j = 0; j < this.n; j++){
			System.out.print("\t" + this.x_positions[i*this.n+j]);
		}
		System.out.println("\n");
	}
        System.out.println("\n");
        System.out.println("Y positions: ");
        for (int i = 0; i < this.n; i++){
		for (int j = 0; j < this.n; j++){
			System.out.print("\t" + this.y_positions[i*this.n+j]);
		}
		System.out.println("\n");
	}
        System.out.println("\n");
	
    }
    
    public int calculateN() {
        this.n = (int) Math.ceil(Math.sqrt((double) this.vertices_list.size()));
        return n * n;
    }
    
    public int getN(){
        return this.N;
    }
    
    public int getn(){
        return this.n;
    }
    
    public float[] getXPositions() {
        return this.x_positions;
    }
    
    public float[] getYPositions() {
        return this.y_positions;
    }
    
    public float[] getXVelocities() {
        return this.x_velocities;
    }
    
    public float[] getYVelocities() {
        return this.y_velocities;
    }
}
