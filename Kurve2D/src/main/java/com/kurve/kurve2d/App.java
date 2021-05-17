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

/**
 * App initialization, entry point
 * @author DanElias
 */
public class App {
    private JCudaSpringForceCalculator jcuda_calculator;
    private String graph_json_url;
    private JSONObject graph_json_object;
    
    public App(String json_url)throws IOException{
        this.graph_json_url = json_url;
        this.graph_json_object = JSONUtils.readJson(this.graph_json_url);
        this.jcuda_calculator = new JCudaSpringForceCalculator("", 1024);
    }
    
    public void start(){
        System.out.println("Now Calculating...");
        //JSONUtils.printJsonObject(this.graph_json_object, "vertices");
        //JSONUtils.printJsonObject(this.graph_json_object, "edges");
        this.jcuda_calculator.calculate();
    }
    
    public static void main(String args[]) throws IOException{
        /*
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
        */
        String url = "src/main/java/com/kurve/kurve2d/data_examples/example.json";
        App app = new App(url);
        app.start();
        System.out.println("Finished");
    }
    
}