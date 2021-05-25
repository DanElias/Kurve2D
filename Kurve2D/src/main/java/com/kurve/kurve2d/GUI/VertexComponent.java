/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.kurve.kurve2d.GUI;

import java.awt.geom.Ellipse2D;

/**
 *
 * @author danie
 */
public class VertexComponent {
    private float x;
    private float y;
    private int WIDTH;
    private int HEIGHT;
    
    public VertexComponent(float x, float y) {
        this.x = x;
        this.y = y;
        this.WIDTH = 10;
        this.HEIGHT = 10;
    }
    
    public void setX(float x) {
        this.x = x;
    }
    
    public void setY(float y) {
        this.y = y;
    }
    
    public Ellipse2D getCircle() {
        return new Ellipse2D.Double(this.x, this.y, this.WIDTH, this.HEIGHT);
    }
}
