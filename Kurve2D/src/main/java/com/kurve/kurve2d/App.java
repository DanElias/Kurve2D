/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.kurve.kurve2d;

import java.io.IOException;
import jcuda.*;
import jcuda.runtime.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import utils.JSONUtils;

import com.kurve.kurve2d.AdjacencyListGraph.ListGraph;
import com.kurve.kurve2d.AdjacencyMatrixGraph.MatrixGraph;
import com.kurve.kurve2d.GUI.Frame;
import javax.swing.ImageIcon;
import javax.swing.JFrame;

/**
 * App initialization, entry point
 * @author DanElias
 */
public class App {
    private Frame frame;
    
    private JCudaSpringForceCalculator jcuda_calculator;
    private String graph_json_url;
    private JSONObject graph_json_object;
    private ListGraph list_graph;
    private MatrixGraph matrix_graph;
    
    public App(String json_url)throws IOException{
        this.graph_json_url = json_url;
        this.graph_json_object = JSONUtils.readJson(this.graph_json_url);
        this.list_graph = new ListGraph(this.graph_json_object);
        this.matrix_graph = new MatrixGraph(this.graph_json_object);
        initializeGUI();
    }
    
    public void start(){
        System.out.println("Now Calculating...");
        //JSONUtils.printJsonObject(this.graph_json_object, "vertices");
        //JSONUtils.printJsonObject(this.graph_json_object, "edges");
        //this.jcuda_calculator.calculate();
        //this.jcuda_calculator.free();
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
        String url = "src/main/java/com/kurve/kurve2d/data_examples/miserables.json";
        App app = new App(url);
        app.start();
        System.out.println("Finished");
    }
    
}