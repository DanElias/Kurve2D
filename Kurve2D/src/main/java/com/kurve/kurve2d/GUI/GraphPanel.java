/**
 * @author DanElias
 * JPanel for the GUI
 */
package com.kurve.kurve2d.GUI;

import com.kurve.kurve2d.Graph.ListGraph;
import com.kurve.kurve2d.Graph.MatrixGraph;
import com.kurve.kurve2d.JCudaSpringForceCalculator;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.util.ArrayList;
import javax.swing.JPanel;

/**
 * @author DanElias
 * JPanel for the GUI
 */
public class GraphPanel extends JPanel{
    private ArrayList<VertexComponent> vertexComponents;
    private ArrayList<EdgeComponent> edgeComponents;
    private MatrixGraph matrix_graph;
    private ListGraph list_graph;
    private JCudaSpringForceCalculator jcuda_calculator;
    
    public GraphPanel(ListGraph list_graph, MatrixGraph matrix_graph) {
        super();
        this.list_graph = list_graph;
        this.matrix_graph = matrix_graph;
        this.vertexComponents = new ArrayList<VertexComponent>();
        this.edgeComponents = new ArrayList<EdgeComponent>();
        setBackground(Color.DARK_GRAY);
        
        int[][] adj_matrix = this.matrix_graph.getAdjacencyMatrix();
        int n = this.matrix_graph.getNumberOfVertices();
        
        // add vertex graphic components
        for (int i = 0; i < this.matrix_graph.getNumberOfVertices(); i++){
            float vertex_x  = this.list_graph.getXPositions()[i];
            float vertex_y  = this.list_graph.getYPositions()[i];
            this.vertexComponents.add(new VertexComponent(vertex_x,vertex_y));
        }
        
        // add edges graphic components
        for (int i = 0; i < n; i++){
            for (int j = i; j < n; j++){
                if (adj_matrix[i][j] == 1) {
                    float vertex_x1  = this.list_graph.getXPositions()[i];
                    float vertex_y1  = this.list_graph.getYPositions()[i];
                    float vertex_x2  = this.list_graph.getXPositions()[j];
                    float vertex_y2  = this.list_graph.getYPositions()[j];
                    this.edgeComponents.add(new EdgeComponent(vertex_x1,vertex_y1, vertex_x2, vertex_y2));
                }
            }
        }
    }
    
    /**
     * Starts the painting of graphics components
     * @param g 
     */
    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setColor(Color.WHITE);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
        update(g2);
    }
    
    /**
     * Repaints panel and its graphic components of every frame
     * @param g2 
     */
    public void update(Graphics2D g2) {
        int n = this.matrix_graph.getNumberOfVertices(); // num of vertices to be drawn
        int edgeListPositions = 0; // traverse the edges UI Components
        
        // Draw the edges first
        for (int i = 0; i < n; i++){
            for (int j = i; j < n; j++){
                if (this.matrix_graph.getAdjacencyMatrix()[i][j] == 1) {
                    float vertex_x1 =
                            this.list_graph.getXPositions()[i] + VertexComponent.getRadius();
                    float vertex_y1 =
                            this.list_graph.getYPositions()[i] + VertexComponent.getRadius();
                    float vertex_x2 =
                            this.list_graph.getXPositions()[j] + VertexComponent.getRadius();
                    float vertex_y2 =
                            this.list_graph.getYPositions()[j] + VertexComponent.getRadius();
                    EdgeComponent edge =
                            this.edgeComponents.get(edgeListPositions);
                    edge.setX1(vertex_x1);
                    edge.setY1(vertex_y1);
                    edge.setX2(vertex_x2);
                    edge.setY2(vertex_y2);
                    g2.setColor(Color.GRAY);
                    g2.draw(edge.getLine());
                    edgeListPositions++;
                }
            }
        }
        
        // Draw the vertices next
        for (int i = 0; i < n; i++){
            VertexComponent vertex = this.vertexComponents.get(i);
            vertex.setX(this.list_graph.getXPositions()[i]);
            vertex.setY(this.list_graph.getYPositions()[i]);
            g2.setColor(Color.WHITE);
            g2.fill(vertex.getCircle());
        }
    }
}
