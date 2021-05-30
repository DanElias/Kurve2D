/**
 * @author DanElias
 * JFrame for the GUI
 */
package com.kurve.kurve2d.GUI;

import com.kurve.kurve2d.Graph.ListGraph;
import com.kurve.kurve2d.Graph.MatrixGraph;
import java.io.IOException;
import javax.swing.JFrame;

/**
 * @author DanElias
 * JFrame for the GUI
 */
public class Frame extends JFrame{
    private final int WIDTH = 1080; //jframe width
    private final int HEIGHT = 720; //jframe width
    private GraphPanel graphPanel; //JPanel to draw the graph
    private GUIThread gui_thread; //Thread that updates JPanel redrawing graph
    
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
