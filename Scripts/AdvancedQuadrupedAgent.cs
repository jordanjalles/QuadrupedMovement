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
    public Transform chest;
    Rigidbody rBody;

    public float minUpVectorDot;
    public float distanceToTouchTarget = 1f;
    public float targetDistanceRange = 18f;

    public enum RewardMode {SeekTarget, StandUp};
    public RewardMode rewardMode;
    
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
    
    private float forceUsedPercent = 0;

    public bool stateBasedModelSwitching = false;

    public Unity.Barracuda.NNModel seekTargetModel; 
    public Unity.Barracuda.NNModel standUpModel;


    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        this.initialMaxSpringForce = maxSpringForce;

        for (int i = 0; i < limbs.Length; i++)
        {
            ConfigurableJoint cJoint = limbs[i].GetComponent<ConfigurableJoint>();
            SetJointDriveMaximumForce(cJoint, maxSpringForce);
        }
    }

    private void Update()
    {
        
        if (stateBasedModelSwitching)
        {
            
            if (rewardMode == RewardMode.StandUp)
            {
                DrawRedGreen(-0.2f);
                if (FallenDownConditions() == false)
                {
                    
                    rewardMode = RewardMode.SeekTarget;
                    this.SetModel("AdvancedQuadrupedBehavior", seekTargetModel);
                    
                }
            }else if (rewardMode == RewardMode.SeekTarget)
            {
                DrawRedGreen(0f);
                if (FallenDownConditions() == true)
                {
                    
                    rewardMode = RewardMode.StandUp;
                    this.SetModel("AdvancedQuadrupedBehavior", standUpModel);
                    
                }
            }
        }
    }

    //Used to reset a training scenario
    //randomize goal location
    //reset position and orientation of agent
    //etc...
    public override void OnEpisodeBegin()
    {
        SetUpTrainingMode();

        if (randomizeMaxSpringForce)
        {
            this.maxSpringForce = this.initialMaxSpringForce * RandomGaussian();
            //Debug.Log(this.maxSpringForce / this.initialMaxSpringForce);
        }

        //todo...make this work properly
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


    private void SetUpTrainingMode()
    {
        if (rewardMode == RewardMode.StandUp)
        {
            if (target.gameObject.activeSelf)
            {
                target.gameObject.SetActive(false);
            }
            ResetLocation();
            //flip around to random rotation
            this.rBody.transform.Rotate(new Vector3(90 + RandomGaussian() * 180, Random.value * 360, 0f));
        }

        if (rewardMode == RewardMode.SeekTarget)
        {
            if (!target.gameObject.activeSelf)
            {
                target.gameObject.SetActive(true);
            }

            if (FallenDownConditions())
            {
                ResetLocation();
                //face random direction
                this.rBody.transform.Rotate(new Vector3(0, Random.value * 360, 0));
            }

            while (DistanceToTarget() < distanceToTouchTarget * 3)
            {
                RandomizeTargetLocation();
            }
        }

    }

    private void ResetLocation()
    {
        this.transform.localPosition = new Vector3(0, 0.07f, 0);
        this.rBody.transform.rotation = Quaternion.identity;
        this.rBody.velocity = Vector3.zero;
        this.rBody.angularVelocity = Vector3.zero;
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

    private bool FallenDownConditions()
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

        this.forceUsedPercent = 0f;
        this.touchGroundOtherThanFeet = false;

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

        if (rewardMode == RewardMode.SeekTarget)
        {
            SeekTargetReward();
        }

        if (rewardMode == RewardMode.StandUp)
        {
            StandUpReward();
        }
    }

    private void StandUpReward()
    {

        if (FallenDownConditions() == false)
        {
            AddReward(10f);
            if (!immortal)
            {
                EndEpisode();
            }
        }

        float reward = 0.0f;
        float upVectorSignal = Mathf.InverseLerp(-1, 1, (Vector3.Dot(this.transform.up, Vector3.up))) * 0.2f;
        reward += Mathf.Pow(upVectorSignal, 4);
        AddReward(reward);
    }

    private void SeekTargetReward() {

        if (FallenDownConditions())
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
        float velocityDeltaFromAverage = (velocityRollingAverage - rBody.velocity).magnitude;
        float movementSmoothness = 1 - (float)System.Math.Tanh(velocityDeltaFromAverage*2f);


        //0 is away from target, 1 is towards target
        float facingTarget = Mathf.InverseLerp(-1, 1, GetFacingTarget());
        //float facingTargetPow2 = Mathf.Pow(facingTarget, 2);
        
        float velocityTowardsTarget = rBody.velocity.magnitude*Vector3.Dot(rBody.velocity.normalized, (target.position - this.transform.position).normalized);
        //velocity towards target limit 1
        float vttl1 = Mathf.Max((float)System.Math.Tanh(velocityTowardsTarget*0.5f), 0);


        float levelHorizon =  Mathf.InverseLerp(minUpVectorDot, 1f, Vector3.Dot(this.transform.up, Vector3.up));


        float reward = 1;

        reward *= facingTarget;
        reward *= Mathf.Pow(vttl1,4);
        reward *= efficiencyRollingAverage;
        //reward *= movementSmoothness;
        //reward *= levelHorizon;

        //DrawRedGreen(vttl1);
        //Debug.Log("vttl1: " + vttl1);


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