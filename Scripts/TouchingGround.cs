using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TouchingGround : MonoBehaviour
{
    public bool touching = false;
    const string k_Target = "ground";
    // Start is called before the first frame update
    /// <summary>
    /// Check for collision with a target.
    /// </summary>
    void OnCollisionEnter(Collision col)
    {
        if (col.transform.CompareTag(k_Target))
        {
            touching = true;
        }
    }

    /// <summary>
    /// Check for end of ground collision and reset flag appropriately.
    /// </summary>
    void OnCollisionExit(Collision other)
    {
        if (other.transform.CompareTag(k_Target))
        {
            touching = false;
        }
    }

}
