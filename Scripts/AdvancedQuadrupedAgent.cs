using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;


public class AdvancedQuadrupedAgent : Agent
{

    public Transform target;
    public float positionSpringPower = 100;
    public float maxSpringForce = 200;
    private float initialMaxSpringForce;
    public Transform[] limbs;
    public float minUpVectorDot;
    public Transform chest;
    Rigidbody rBody;
    public float distanceToTouchTarget = 1f;
    public float targetDistanceRange = 18f;
    public bool immortal = false;
    public bool randomizeMaxSpringForce = false;
    public bool randomizeLegScales = false;
    public float frontScale = 0.75f;
    public float backScale = 0.75f;
    private float efficiencyRollingAverage = 0;
    private float efficiencyRollingAverageDepth = 10; 
    private Vector3 velocityRollingAverage = Vector3.zero;
    private float velocityRollingAverageDepth = 20;
    private bool touchGroundOtherThanFeet = false;


    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        this.initialMaxSpringForce = maxSpringForce;

        for (int i = 0; i < limbs.Length; i++)
        {
            ConfigurableJoint cJoint = limbs[i].GetComponent<ConfigurableJoint>();
            SetJointDriveMaximumForce(cJoint, maxSpringForce);
        }

        float previousDistanceToTarget = Vector3.Distance(this.transform.localPosition, target.localPosition);

    }

    //Used to reset a training scenario
    //randomize goal location
    //reset position and orientation of agent
    //etc...
    public override void OnEpisodeBegin()
    {
        RandomizeTargetLocation();
        while (DistanceToTarget() < distanceToTouchTarget * 3)
        {
            RandomizeTargetLocation();
        }

        if (DeathConditions())
        {
            this.rBody.transform.rotation = Quaternion.identity;
            this.rBody.velocity = Vector3.zero;
            this.rBody.angularVelocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.07f, 0);
            this.rBody.transform.Rotate(new Vector3(0, Random.value * 360, 0));
        }

        if (randomizeMaxSpringForce)
        {
            this.maxSpringForce = this.initialMaxSpringForce * RandomGaussian();
            //Debug.Log(this.maxSpringForce / this.initialMaxSpringForce);
        }

        if (randomizeLegScales)
        {
            Transform backRight = limbs[0].transform.parent;
            Transform backLeft = limbs[2].transform.parent;
            Transform frontRight = limbs[4].transform.parent;
            Transform frontLeft = limbs[6].transform.parent;

            //float frontScale = RandomGaussian()*2f + 0.5f;
            //float backScale = RandomGaussian()*2f + 0.5f;
            frontScale = 1.2f;
            backScale = 1.2f;

            ResizeLimbParent(backRight, backScale);
            ResizeLimbParent(backLeft, backScale);
            ResizeLimbParent(frontRight, frontScale);
            ResizeLimbParent(frontLeft, frontScale);

        }
    }

    private void ResizeLimbParent(Transform limbParent, float newScale)
    {
        //assuming uniform scale in all dimensions for now
        
        float oldScale = limbParent.localScale.x;
        /*
        limbParent.localScale = (Vector3.one * newScale);

        ConfigurableJoint[] joints = limbParent.GetComponentsInChildren<ConfigurableJoint>();
        */
        foreach (Transform limb in limbParent.GetComponentsInChildren<Transform>())
        {
            limb.localScale *= newScale / oldScale;
            limb.GetComponentInChildren<ConfigurableJoint>().autoConfigureConnectedAnchor = false;
        }
    }

    private void RandomizeTargetLocation()
    {
        target.localPosition = new Vector3(Random.value * targetDistanceRange - targetDistanceRange/2, 0, Random.value * targetDistanceRange - targetDistanceRange/2);
    }

    private bool DeathConditions()
    {
        return (touchGroundOtherThanFeet || Vector3.Dot(this.transform.up, Vector3.up) < minUpVectorDot || chest.GetComponent<TouchingGround>().touching);
    }

    //total observation size 32
    public override void CollectObservations(VectorSensor sensor)
    {
        
        
        //target position vec3
        sensor.AddObservation(this.transform.InverseTransformPoint(target.position).normalized);
        
        //body momentum vec3
        sensor.AddObservation(rBody.velocity);

        //body angular velocity vec3
        sensor.AddObservation(rBody.angularVelocity);

        //body rotation vec3
        sensor.AddObservation(rBody.transform.rotation);

        //facing target
        sensor.AddObservation(GetFacingTarget());

        //limb positions vec3 * 12
        //limb rotation vec3 * 12
        //limb touchign ground bool * 12
        foreach (Transform limb in limbs)
        {
            //limb attributes relative to body
            sensor.AddObservation(this.transform.InverseTransformPoint(limb.position));
            sensor.AddObservation(limb.GetComponent<ConfigurableJoint>().slerpDrive.maximumForce / this.maxSpringForce);
            sensor.AddObservation(limb.transform.localRotation);
            sensor.AddObservation(this.transform.InverseTransformDirection(limb.GetComponent<Rigidbody>().velocity));
            sensor.AddObservation(this.transform.InverseTransformDirection(limb.GetComponent<Rigidbody>().angularVelocity));
            sensor.AddObservation(limb.GetComponent<TouchingGround>().touching);
        }
        
        sensor.AddObservation(this.maxSpringForce);
        sensor.AddObservation(frontScale);
        sensor.AddObservation(backScale);
        
    }

    //execute the actions given from the brain
    //update rewards
    public override void OnActionReceived(float[] vectorAction)
    {
        Vector3 controlSignal = Vector3.zero;
        float minRotation;
        float maxRotation;
        float forceUsedPercent = 0f;
        touchGroundOtherThanFeet = false;

        for (int i = 0; i < limbs.Length; i++) {
            ConfigurableJoint cJoint = limbs[i].GetComponent<ConfigurableJoint>();
            minRotation = cJoint.lowAngularXLimit.limit;
            maxRotation = cJoint.highAngularXLimit.limit;
            controlSignal.x = Mathf.Lerp(minRotation, maxRotation, Mathf.InverseLerp(-1, 1, vectorAction[(i * 4)]));

            minRotation = -cJoint.angularYLimit.limit;
            maxRotation = cJoint.angularYLimit.limit;
            controlSignal.y = Mathf.Lerp(minRotation, maxRotation, Mathf.InverseLerp(-1, 1, vectorAction[(i * 4) + 1]));

            minRotation = -cJoint.angularZLimit.limit;
            maxRotation = cJoint.angularZLimit.limit;
            controlSignal.z = Mathf.Lerp(minRotation, maxRotation, Mathf.InverseLerp(-1, 1, vectorAction[(i * 4) + 2]));

            float forceSignal = Mathf.InverseLerp(-1, 1, vectorAction[i * 4 + 3]);
            forceUsedPercent += forceSignal / limbs.Length;

            float maximumForce = maxSpringForce * forceSignal;
            SetJointDriveMaximumForce(cJoint, maximumForce);

            bool limbTouchingGround = limbs[i].GetComponent<TouchingGround>().touching;
            if (limbTouchingGround && limbs[i].name != "lower")
            {
                touchGroundOtherThanFeet = true;
            }

            cJoint.targetRotation = Quaternion.Euler(controlSignal);
        }


        if (DeathConditions())
        {            
            SetReward(-1.0f);
            if (!immortal)
            {
                EndEpisode();
            }
        }

        float efficiency = (1 - forceUsedPercent);
        efficiencyRollingAverage = (efficiencyRollingAverage + (efficiency / efficiencyRollingAverageDepth)) / (1 + 1f / efficiencyRollingAverageDepth);
        //Debug.Log("e:" + efficiency + " era" + efficiencyRollingAverage.ToString());

        velocityRollingAverage = (velocityRollingAverage + (rBody.velocity / velocityRollingAverageDepth)) / (1 + 1f / velocityRollingAverageDepth);
        //Debug.Log("v:" + rBody.velocity.ToString() + "vra" + velocityRollingAverage.ToString());
        float velocityDelta = (velocityRollingAverage - rBody.velocity).magnitude;
        float movementSmoothness = 1 - (float)System.Math.Tanh(velocityDelta*2f);
        //DrawRedGreen(movementSmoothness*2-1);
        //Debug.Log("ms: " + movementSmoothness.ToString());


        //0 is away from target, 1 is towards target
        float facingTarget = Mathf.InverseLerp(-1, 1, GetFacingTarget());
        //float facingTargetPow2 = Mathf.Pow(facingTarget, 2);
        
        float velocityTowardsTarget = rBody.velocity.magnitude*Vector3.Dot(rBody.velocity.normalized, (target.position - this.transform.position).normalized);
        //velocity towards target limit 1
        float vttl1 = Mathf.Max((float)System.Math.Tanh(velocityTowardsTarget*0.5f), 0);
        //Debug.Log("vttl1: " + vttl1.ToString());
        //DrawRedGreen(vttl1);

        float levelHorizon =  Mathf.InverseLerp(minUpVectorDot, 1f, Vector3.Dot(this.transform.up, Vector3.up));
        
        //DrawRedGreen(levelHorizon*2-1);
        //Debug.Log(levelHorizon);



        float reward = 1;

        reward *= facingTarget;
        reward *= Mathf.Pow(vttl1,4);
        reward *= efficiencyRollingAverage;
        //reward *= movementSmoothness;
        //reward *= levelHorizon;

        DrawRedGreen(vttl1);
        Debug.Log("vttl1: " + vttl1);


        AddReward(reward); //reward for moving towards goal
        //Debug.Log("reward:" + reward);
        




        //Debug.Log(distanceToTarget);
        if (DistanceToTarget() < distanceToTouchTarget)
        {
            SetReward(1.0f);
            EndEpisode();
        }


    }

    private float DistanceToTarget()
    {
        return Vector3.Distance(this.transform.localPosition, target.localPosition);
    }
    private void DrawRedGreen(float reward)
    {

        //Debug.Log("draw reward: "+ transform.GetComponentsInChildren<Renderer>().ToString());
        foreach (Renderer rend in transform.GetComponentsInChildren<Renderer>())
        {
            Color c;
            if (reward < 0)
            {
                c = Color.Lerp(Color.red, Color.white, Mathf.InverseLerp(-1, 0, reward));
            }
            else
            {
                c = Color.Lerp(Color.white, Color.green, Mathf.InverseLerp(0, 1, reward));
            }

            if (rend.material.name == "BodyMatV2 (Instance)") { 
                //rend.material.SetColor("_EmissionColor", c);
                rend.material.SetColor("_BaseColor", c);
            }
        }


    }

    private void SetJointDriveMaximumForce(ConfigurableJoint cJoint, float newMaximumForce)
    {
        JointDrive jDrive = new JointDrive();
        jDrive.maximumForce = newMaximumForce;
        jDrive.positionSpring = this.positionSpringPower;
        cJoint.slerpDrive = jDrive;
    }

    public override void Heuristic(float[] actionsOut)
    {
        for (int i = 0; i < actionsOut.Length; i++)
        {
            float v = Mathf.InverseLerp(-1, 1, Input.GetAxis("Horizontal"));
            float h = Mathf.InverseLerp(-1, 1, Input.GetAxis("Vertical"));
            //actionsOut[i] = (i % 2 == 0) ? v : h;
        }
    }

    private float limit1(float number, float halfWayMark)
    {
        return 1 - (1 / (number / halfWayMark + 1));
    }

    private float limit1(float number)
    {
        return limit1(number, 0.5f);
    }

    public float GetComplacency()
    {
        return  Mathf.Pow(1 - (limit1(rBody.velocity.magnitude, 0.25f)), 2);
    }

    public float GetFacingTarget()
    {
        return Vector3.Dot(this.transform.forward, (this.transform.position - target.position).normalized);
    }

    public float RandomGaussian()
    {
        //average of three random values approximates gaussian sample
        return (Random.value + Random.value + Random.value) / 3f;
    }
}