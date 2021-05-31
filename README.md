# **Kurve - 2D Force Directed Graph Visualizer**
### **_For business, science and creators_**

*Kurve Graph Visualizer is an open source project that lets users visualize their graph data in 2D space.*

Graph visualizations ideas include: 
- LAN and WAN networks
- User connections in social networks
- Vector Space Model for relations between words or images
- Artificial Intelligence Neural Networks
- Relations between characters in a play or novel
- Transport infrastructure between cities.
- Etc.

## Author
- Daniel Elias Becerra - daniel.eliasbecerra98@gmail.com

## Version:
- 1.0.0

## Technologies:
- Java
- JCuda
- Cuda

## User Features:
- Visualize graphs in 3D space
- Customize your graph's vertices with colors and icons
- 3 options of graph visualizations: Simple Graph, Les Miserables Character Connections, Networks.

## Technical Features:
- Implementation of the [Eades Force Directed Graph algorithm](http://cs.brown.edu/people/rtamassi/gdhandbook/chapters/force-directed.pdf) which treats the graph as a mechanical system with springs or electrical forces.
- Connected vertices attract each other, while disconnected ones repell one another. All these using the Cannon js physics engine.
- The app accepts any kind of json in the format described later on. At the moment the app has only 5 data visualizations.


## JSON Format 
- for Graph Data
```json
{
    "vertices": [
        {
            "id": "A",
            "name": "A"
        },
        {
            "id": "B",
            "name": "B"
        },
    ],
    "edges": [
        {
            "source": "A",
            "target": "B"
        },
    ]
}
```


