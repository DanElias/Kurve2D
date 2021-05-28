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
    private static final float RADIUS = (float) 3.5;
    
    public VertexComponent(float x, float y) {
        this.x = x;
    }
    
    public static float getRadius() {
        return RADIUS;
    }
    
    public void setX(float x) {
        this.x = x;
    }
    
    public void setY(float y) {
        this.y = y;
    }
    
    public Ellipse2D getCircle() {
        return new Ellipse2D.Double(this.x, this.y, this.RADIUS * 2, this.RADIUS * 2);
    }
}
