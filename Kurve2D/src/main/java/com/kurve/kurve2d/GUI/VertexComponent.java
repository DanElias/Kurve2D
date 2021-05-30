/**
 * UI Vertex component
 * @author DanElias
 */
package com.kurve.kurve2d.GUI;

import java.awt.geom.Ellipse2D;

/**
 * UI Vertex component
 * @author DanElias
 */
public class VertexComponent {
    private float x; // coordinate x
    private float y; // cordinate y
    private static final float RADIUS = (float) 3.5; // ellipse radius
    
    /**
     * @author DanElias
     * Set starting (x,y) coordinate for the ellipse
     * @param x
     * @param y 
     */
    public VertexComponent(float x, float y) {
        this.x = x;
        this.y = y;
    }
    
    /**
     * @author DanElias
     * @return redius
     */
    public static float getRadius() {
        return RADIUS;
    }
    
    /**
     * @author DanElias
     * Set the nex x coordinate
     * @param x 
     */
    public void setX(float x) {
        this.x = x;
    }
    
    /**
     * @author DanElias
     * Set the nex y coordinate
     * @param y 
     */
    public void setY(float y) {
        this.y = y;
    }
    
    /**
     * @author DanElias
     * @return circle drawing
     */
    public Ellipse2D getCircle() {
        return new Ellipse2D.Double(this.x, this.y, this.RADIUS * 2, this.RADIUS * 2);
    }
}
