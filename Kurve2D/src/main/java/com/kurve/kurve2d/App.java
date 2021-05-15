/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.kurve.kurve2d;

import jcuda.*;
import jcuda.runtime.*;

/**
 *
 * @author danie
 */
public class App {
    public static void main(String args[])
    {
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
        
        System.out.println("RRR");
    }
}


