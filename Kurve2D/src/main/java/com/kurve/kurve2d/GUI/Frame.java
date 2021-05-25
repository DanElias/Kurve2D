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
import javax.swing.JFrame;

/**
 *
 * @author danie
 */
public class Frame extends JFrame{
    private final int WIDTH = 1080;
    private final int HEIGHT = 720;
    private GraphPanel graphPanel;
    private GUIThread gui_thread;
    
    public Frame(ListGraph list_graph, MatrixGraph matrix_graph) throws IOException{
        super();
        setTitle("Kurve2D");
        setSize(this.WIDTH, this.HEIGHT);
        setLocationRelativeTo(null); // center in window
        setResizable(false);
        this.graphPanel = new GraphPanel(list_graph, matrix_graph);
        add(this.graphPanel);
        this.gui_thread = new GUIThread(this.graphPanel, list_graph, matrix_graph);
        this.gui_thread.start();
    }
}
