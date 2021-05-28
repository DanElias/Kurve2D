/**
 * Main App
 * @author DanElias
 */
package com.kurve.kurve2d;

import java.io.IOException;

import org.json.simple.JSONObject;

import utils.JSONUtils;

import com.kurve.kurve2d.AdjacencyListGraph.ListGraph;
import com.kurve.kurve2d.AdjacencyMatrixGraph.MatrixGraph;
import com.kurve.kurve2d.GUI.Frame;
import javax.swing.ImageIcon;
import javax.swing.JFrame;

/**
 * App initialization, entry point and starts GUI
 * @author DanElias
 */
public class App {
    private Frame frame; //JFrame
    private String graph_json_url; // data
    private JSONObject graph_json_object;
    private ListGraph list_graph; // Graph representations
    private MatrixGraph matrix_graph;
    
    public App(String json_url)throws IOException{
        // Gets data and initializes graph's data
        this.graph_json_url = json_url;
        this.graph_json_object = JSONUtils.readJson(this.graph_json_url);
        this.list_graph = new ListGraph(this.graph_json_object);
        this.matrix_graph = new MatrixGraph(this.graph_json_object);
        // starts the animation/force directed graph simulation
        System.out.println(this.matrix_graph.getNumberOfVertices());
        System.out.println(this.matrix_graph.getNumberOfEdges());
        initializeGUI();
    }
    
    public void initializeGUI() throws IOException {
        // Set the GUI
        this.frame = new Frame(this.list_graph, this.matrix_graph);
        ImageIcon logo = new ImageIcon("src/main/java/com/kurve/kurve2d/GUI/assets/images/logo.png");
        this.frame.setIconImage(logo.getImage());
        this.frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.frame.setVisible(true);
    }
    
    public static void main(String args[]) throws IOException{
        String url = "src/main/java/com/kurve/kurve2d/data_examples/blocks.json";
        App app = new App(url);
        System.out.println("Finished");
    }
    
}