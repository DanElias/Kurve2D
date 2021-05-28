/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.kurve.kurve2d.GUI;

import java.awt.geom.Line2D;

/**
 *
 * @author danie
 */
public class EdgeComponent {
    private float x1;
    private float y1;
    private float x2;
    private float y2;
    
    public EdgeComponent(float x1, float y1, float x2, float y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }
    
    public void setX1(float x1) {
        this.x1 = x1;
    }
    
    public void setY1(float y1) {
        this.y1 = y1;
    }
    
    public void setX2(float x2) {
        this.x2 = x2;
    }
    
    public void setY2(float y2) {
        this.y2 = y2;
    }
    
    public Line2D getLine() {
        return new Line2D.Double(this.x1, this.y1, this.x2, this.y2);
    }
}
