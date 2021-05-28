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
    private static final int NO_DELAYS_PER_YIELD = 16;
    /* Number of frames with a delay of 0 ms before the
    animation thread yields to other running threads. */
    private static int MAX_FRAME_SKIPS = 1;
    private static int M = 10000;
    
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
    
    private void updatePositions() {
        this.jcuda_calculator2.calculate();
        this.graph_panel.repaint();
    }
    
    @Override
    public void run(){
        int t_minus_M = 0;
        long beforeTime, afterTime, timeDiff, sleepTime;
        long period = 1000000000/85; //period = 1000/desiredFPS
        long overSleepTime = 0L;
        int noDelays = 0;
        long excess = 0L;
        beforeTime = java.lang.System.nanoTime();
        
        long startTime = System.nanoTime();
        while(t_minus_M < M) {
            t_minus_M++;
            updatePositions();
                
            afterTime = java.lang.System.nanoTime();
            timeDiff = afterTime - beforeTime;
            sleepTime = (period - timeDiff) - overSleepTime; //time left in this loop
            if (sleepTime > 0){
                try {
                    Thread.sleep(sleepTime/1000000L); // nano -> ms
                    
                } catch (InterruptedException ex) {
                    System.out.println(ex);
                }
                overSleepTime = (java.lang.System.nanoTime() - afterTime) - sleepTime;
            } else {
                excess -= sleepTime;  // store excess time value
                overSleepTime = 0L;
                if (++noDelays >= NO_DELAYS_PER_YIELD) {
                  Thread.yield();   // give another thread a chance to run
                  noDelays = 0;
                }
            }
            beforeTime = java.lang.System.nanoTime();
            /* If frame animation is taking too long, update the game state
               without rendering it, to get the updates/sec nearer to
               the required FPS. */
            int skips = 0;

            while((excess > period) && (skips <= MAX_FRAME_SKIPS)) {
                excess -= period;
                updatePositions();
                skips++;
                t_minus_M++;
            } //end of while2
        }
        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;
        System.out.println("Execution time in nanoseconds: " + timeElapsed);
        System.out.println("Execution time in milliseconds: " + timeElapsed / 1000000);
    }
}
