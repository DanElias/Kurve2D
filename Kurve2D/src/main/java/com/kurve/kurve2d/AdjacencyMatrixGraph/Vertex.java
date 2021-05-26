/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.kurve.kurve2d.AdjacencyMatrixGraph;

import com.kurve.kurve2d.AdjacencyListGraph.*;
import java.util.ArrayList;

/**
 *
 * @author danie
 */
public class Vertex {
    private int id;
    private String original_id;
    private String value;
    private ArrayList<Integer> adjacent_vertices;
    private ArrayList<Integer> nonadjacent_vertices;
    
    public Vertex(int id, String original_id, String value) {
        this.id = id;
        this.original_id = original_id;
        this.value = value;
        this.adjacent_vertices = new ArrayList<Integer>();
        this.nonadjacent_vertices = new ArrayList<Integer>();
    }
    
    public void addAdjacentVertex(int vertex_id) {
        this.adjacent_vertices.add(vertex_id);
    }
    
    public void addNonadjacentVertex(int vertex_id) {
        this.nonadjacent_vertices.add(vertex_id);
    }
    
    public void setId(int id) {
        this.id = id;
    }
    
    public int getId() {
        return this.id;
    }
    
    public void setOriginalId(String id) {
        this.original_id = id;
    }
    
    public String getOriginalId() {
        return this.original_id;
    }
    
    public void setValue(String value) {
        this.value = value;
    }
    
    public String getValue() {
        return this.value;
    }
    
    public void setAdjacentVertices(ArrayList<Integer> adjacent_vertices) {
        this.adjacent_vertices = new ArrayList<>(adjacent_vertices);
    }
    
    public ArrayList<Integer> getAdjacentVertices() {
        return this.adjacent_vertices;
    }
    
    public void setNonadjacentVertices(ArrayList<Integer> nonadjacent_vertices) {
        this.nonadjacent_vertices = new ArrayList<>(nonadjacent_vertices);
    }
    
    public ArrayList<Integer> getNonadjacentVertices() {
        return this.nonadjacent_vertices;
    }
    
}
