/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.kurve.kurve2d.GUI;

import com.kurve.kurve2d.AdjacencyListGraph.ListGraph;
import com.kurve.kurve2d.AdjacencyMatrixGraph.MatrixGraph;
import com.kurve.kurve2d.JCudaSpringForceCalculator;
import java.io.IOException;

/**
 *
 * @author danie
 */
public class GUIThread extends Thread{
    private GraphPanel graph_panel;
    private MatrixGraph matrix_graph;
    private ListGraph list_graph;
    private JCudaSpringForceCalculator jcuda_calculator;
    private JCudaSpringForceCalculator jcuda_calculator2;
    
    public GUIThread(GraphPanel graph_panel, ListGraph list_graph, MatrixGraph matrix_graph) throws IOException{
        this.graph_panel = graph_panel;
        this.list_graph = list_graph;
        this.matrix_graph = matrix_graph;
        initializeJCudaSpringForceCalculator();
    }
    
    public void initializeJCudaSpringForceCalculator() throws IOException {
        try {
        this.jcuda_calculator2 = new JCudaSpringForceCalculator(
                "", // ptx filename url
                this.matrix_graph.getNumberOfVertices(), // num of vertices * num of vertices
                this.list_graph.getN(), // n * n = size of x/y positions matrix
                this.matrix_graph.getLinearAdjacencyMatrix(), // adjacency matrix graph
                this.list_graph.getXPositions(),
                this.list_graph.getYPositions(),
                this.list_graph.getXVelocities(),
                this.list_graph.getYVelocities()
        );
        } catch (IOException ex) {
             System.out.println(ex);
        }
    }
    
    @Override
    public void run(){
        
        while(true) {
            try {
                Thread.sleep(1000);
                //this.jcuda_calculator2.calculate();
                this.graph_panel.repaint();
            } catch (InterruptedException ex) {
                System.out.println(ex);
            }
        }
    }
}
