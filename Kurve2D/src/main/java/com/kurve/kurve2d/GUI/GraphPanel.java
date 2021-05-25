/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.kurve.kurve2d.GUI;

import com.kurve.kurve2d.AdjacencyListGraph.ListGraph;
import com.kurve.kurve2d.AdjacencyMatrixGraph.MatrixGraph;
import com.kurve.kurve2d.JCudaSpringForceCalculator;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Shape;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import javax.swing.JPanel;

/**
 *
 * @author danie
 */
public class GraphPanel extends JPanel{
    private ArrayList<VertexComponent> vertexComponents;
    private MatrixGraph matrix_graph;
    private ListGraph list_graph;
    private JCudaSpringForceCalculator jcuda_calculator;
    
    public GraphPanel(ListGraph list_graph, MatrixGraph matrix_graph) {
        super();
        this.list_graph = list_graph;
        this.matrix_graph = matrix_graph;
        this.jcuda_calculator = jcuda_calculator;
        this.vertexComponents = new ArrayList<VertexComponent>();
        setBackground(Color.DARK_GRAY);
        for (int i = 0; i < this.matrix_graph.getNumberOfVertices(); i++){
            float vertex_x  = this.list_graph.getXPositions()[i];
            float vertex_y  = this.list_graph.getYPositions()[i];
            this.vertexComponents.add(new VertexComponent(vertex_x,vertex_y));
        }
    }
    
    
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setColor(Color.WHITE);
        Shape circleShape = new Ellipse2D.Double(50, 50, 50, 50);
        draw(g2);
        update(g2);
    }
    
    public void draw(Graphics2D g2) {
        for (VertexComponent vertexComponent : this.vertexComponents){
            g2.fill(vertexComponent.getCircle());
        }
    }
    
    public void update(Graphics2D g2) {
        System.out.println("UPDATE");
        for (int i = 0; i < this.matrix_graph.getNumberOfVertices(); i++){
            VertexComponent vertex = this.vertexComponents.get(i);
            vertex.setX(this.list_graph.getXPositions()[i]);
            vertex.setY(this.list_graph.getXPositions()[i]);
            g2.fill(this.vertexComponents.get(i).getCircle());
        }
    }
}
