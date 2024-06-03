// Code copied from https://discussions.unity.com/t/display-coordinate-system/172516/3 by LeEHil
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class drawCoordinateSystem : MonoBehaviour
{
    public void OnDrawGizmosSelected()
    {
        // draw y axis with arrow
        Gizmos.color = Color.green;
        Gizmos.DrawLine(transform.position, (transform.position + (2 * transform.up)));
        Cone.DrawCone(transform.position + (2 * transform.up), transform.up, 0.1f);

        // draw x axis with arrow
        Gizmos.color = Color.red;
        Gizmos.DrawLine(transform.position, (transform.position + (2 * transform.right)));
        Cone.DrawCone(transform.position + (2 * transform.right), transform.right, 0.1f);

        // draw z axis with arrow
        Gizmos.color = Color.blue;
        Gizmos.DrawLine(transform.position, (transform.position + (2 * transform.forward)));
        Cone.DrawCone(transform.position + (2 * transform.forward), transform.forward, 0.1f);
    }
    private static class Cone
    {
        private static Vector3[] newVertices = new Vector3[]
        {
            new Vector3(-0.0f, 1.0f, 0.0f),
            new Vector3(0.2f, 0.0f, 0.4f),
            new Vector3(0.5f, 0.0f, 0.0f),
            new Vector3(0.0f, 0.0f, 0.0f),
            new Vector3(-0.3f, 0.0f, 0.4f),
            new Vector3(-0.5f, 0.0f, 0.0f),
            new Vector3(-0.2f, 0.0f, -0.4f),
            new Vector3(0.2f, 0.0f, -0.4f),
        };
        private static Vector3[] newNormals = new Vector3[]
        {
            new Vector3(0,1,0),
            new Vector3(1,0,1),
            new Vector3(1,0,0),
            new Vector3(0,-1,0),
            new Vector3(-1,0,1),
            new Vector3(-1,0,0),
            new Vector3(-1,0,-1),
            new Vector3(1,0,-1),
        };
        private static int[] newTriangles = new int[]
        {
            0, 1, 2, 2, 1, 3, 0, 4, 1, 1, 4, 3, 0, 5, 4, 4, 5, 3, 0, 6, 5, 5, 6, 3, 0, 7, 6, 6, 7, 3, 0, 2, 7, 7, 2, 3
        };

        private static Mesh mesh;

        static Cone()
        {
            mesh = new Mesh();
            mesh.vertices = newVertices;
            mesh.triangles = newTriangles;
            mesh.normals = newNormals;
        }

        public static void DrawCone(Vector3 position, Vector3 rotation, float scale)
        {
            Gizmos.DrawMesh(mesh, position, Quaternion.FromToRotation(Vector3.up, rotation), new Vector3(scale, 2 * scale, scale));
        }
    }

    // public GameObject target;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        OnDrawGizmosSelected();
    }
}