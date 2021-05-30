/**
 * Thread that repaints the JPanel on every frame
 * This Thread also calls the Spring Force Calculator on its parallel or iterative version
 * @author DanElias
 */
package com.kurve.kurve2d.GUI;

import com.kurve.kurve2d.Graph.ListGraph;
import com.kurve.kurve2d.Graph.MatrixGraph;
import com.kurve.kurve2d.IterativeSpringForceCalculator;
import com.kurve.kurve2d.JCudaSpringForceCalculator;
import java.io.IOException;

/**
 * Thread that repaints the JPanel on every frame
 * This Thread also calls the Spring Force Calculator on its parallel or iterative version
 * @author DanElias
 */
public class GUIThread extends Thread{
    private GraphPanel graph_panel;
    private MatrixGraph matrix_graph;
    private ListGraph list_graph;
    private JCudaSpringForceCalculator jcuda_calculator;
    private IterativeSpringForceCalculator iterative_calculator;
    private static final int NO_DELAYS_PER_YIELD = 16;
    /* Number of frames with a delay of 0 ms before the
    animation thread yields to other running threads. */
    private static int MAX_FRAME_SKIPS = 1;
    // M is defined by the algorithm of Spring Force Directed Graphs
    private static int M = 100; 
    
    /**
     * Initializes object
     * @author DanElias
     * @param graph_panel
     * @param list_graph
     * @param matrix_graph
     * @throws IOException 
     */
    public GUIThread(GraphPanel graph_panel, ListGraph list_graph, MatrixGraph matrix_graph) throws IOException{
        this.graph_panel = graph_panel;
        this.list_graph = list_graph;
        this.matrix_graph = matrix_graph;
        initializeJCudaSpringForceCalculator();
        initializeIterativeSpringForceCalculator();
    }
    
    /**
     * Initialize the JCuda Calculator
     * @author DanElias
     * @throws IOException 
     */
    public void initializeJCudaSpringForceCalculator() throws IOException {
        try {
            this.jcuda_calculator = new JCudaSpringForceCalculator(
                    "", // ptx filename url
                    this.matrix_graph.getNumberOfVertices(), //  num of vertices = size of x/y positions matrix
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
    
    /**
     * @author DanElias
     * Initialize the Iterative Calculator
     */
    public void initializeIterativeSpringForceCalculator(){
        this.iterative_calculator = new IterativeSpringForceCalculator(
                this.matrix_graph.getNumberOfVertices(), // num of vertices = size of x/y positions matrix
                this.matrix_graph.getLinearAdjacencyMatrix(), // adjacency matrix graph
                this.list_graph.getXPositions(),
                this.list_graph.getYPositions(),
                this.list_graph.getXVelocities(),
                this.list_graph.getYVelocities()
        );   
    }
    
    /**
     * @author DanElias
     * Choose between the parallel or iterative calculator
     * repaint the JPanel
     */
    private void updatePositions() {
        this.jcuda_calculator.calculate();
        /*
        this.iterative_calculator.calculate(
                this.matrix_graph.getNumberOfVertices(), // num of vertices
                this.matrix_graph.getLinearAdjacencyMatrix(), // adjacency matrix graph
                this.list_graph.getXPositions(),
                this.list_graph.getYPositions(),
                this.list_graph.getXVelocities(),
                this.list_graph.getYVelocities());
        */
        this.graph_panel.repaint();
    }
    
    /**
     * @author DanElias
     * Updates every frame
     * All variables are for the app to draw similarly independtly of the hardware
     */
    @Override
    public void run(){
        // M is defined by the algorithm of Spring Force Directed Graphs
        int t_minus_M = 0;
        long beforeTime, afterTime, timeDiff, sleepTime;
        long period = 1000000000/85; //period = 1000/desiredFPS
        long overSleepTime = 0L;
        int noDelays = 0;
        long excess = 0L;
        beforeTime = java.lang.System.nanoTime();
        
        // measure the time it takes to run M iterations
        long startTime = System.nanoTime(); 
        while(t_minus_M < M) {
            t_minus_M++;
            
            updatePositions(); // UPDATE the Graph positions with the FDG algorithm
            
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
                excess -= sleepTime; // store excess time value
                overSleepTime = 0L;
                if (++noDelays >= NO_DELAYS_PER_YIELD) {
                  Thread.yield(); // give another thread a chance to run
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
                
                updatePositions(); // UPDATE the Graph positions with the FDG algorithm
                
                skips++;
                t_minus_M++;
            }
        }
        // Measurements for running time of M iterations
        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;
        /*
        System.out.println("Execution time in nanoseconds: " + ((float) timeElapsed));
        System.out.println("Execution time in milliseconds: " + ((float) timeElapsed / 1000000.0));
        System.out.println("Execution time in seconds: " + ((float) timeElapsed / 1000000000.0));
        */
        // free the memory allocated in the GPU device
        this.jcuda_calculator.free();
    }
}
