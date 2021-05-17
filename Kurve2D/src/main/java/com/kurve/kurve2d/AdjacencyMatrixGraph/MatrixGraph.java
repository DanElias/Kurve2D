package com.kurve.kurve2d.AdjacencyMatrixGraph;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import utils.Utils;

import java.lang.Math;
import java.util.HashSet;
import utils.JSONUtils;

import com.kurve.kurve2d.AdjacencyMatrixGraph.Edge;
import com.kurve.kurve2d.AdjacencyMatrixGraph.Vertex;

/**
 *
 * @author DanElias
 */
public class MatrixGraph {
    private HashMap<String, Integer> vertices_ids; // Original id - mat index mapping
    private ArrayList<Edge> edges;
    public int[][] adjacency_matrix;
    private int N; // CUDA problem size
    private int n; // number of vertices
    private float[][] x_positions_matrix;
    private float[][] y_positions_matrix;
    private List<JSONObject> vertices_list; // Json vertices
    private List<JSONObject> edges_list; // Json edges
    
    public MatrixGraph(JSONObject graph_json){
        this.vertices_ids = new HashMap<String, Integer>();
        this.edges = new ArrayList<Edge>();
        
        Object json_vertices = graph_json.get("vertices");
        this.vertices_list = JSONUtils.objectToJSONObjectArrayList(json_vertices);
        
        Object json_edges = graph_json.get("edges");
        this.edges_list = JSONUtils.objectToJSONObjectArrayList(json_edges);
        
        this.N = calculateN();
        
        this.x_positions_matrix = new float[this.n][this.n];
        this.y_positions_matrix = new float[this.n][this.n];
        this.adjacency_matrix = new int[this.N][this.N];
        
        setVertices();
        setEdges();
    }
    
    private void setVertices() {
        int index = 0;
        for (JSONObject vertex : this.vertices_list){
            String vertex_id = vertex.get("id").toString();
            this.vertices_ids.put(vertex_id, index);
            index++;
        }
    }
    
    private void setEdges() {
        for (JSONObject edge : this.edges_list){
            String source = edge.get("source").toString();
            String target = edge.get("target").toString();
            int source_vertex_id = this.vertices_ids.get(source);
            int target_vertex_id = this.vertices_ids.get(target);
            addEdge(source_vertex_id, target_vertex_id);
        }
    }
    
    public void addEdge(int source, int target){
        adjacency_matrix[source][target] = 1;
        adjacency_matrix[source][target] = 1;
    }
    
    public int calculateN() {
        this.n = this.vertices_list.size();
        return n * n;
    }
    
    public int getN(){
        return this.N;
    }
    
    public int getn(){
        return this.n;
    }
    
    public int[][] getAdjacencyMatrix() {
        return this.adjacency_matrix;
    }
}
