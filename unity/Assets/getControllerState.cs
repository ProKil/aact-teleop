// Code from https://github.com/abitha-thankaraj/xarm-vr/blob/main/Assets/Scripts/ControllerState.cs

using UnityEngine;
using Newtonsoft.Json;
public class ControllerState
{
    private OVRInput.Controller leftController;
    private OVRInput.Controller rightController;
    private GameObject leftControllerObject;
    private GameObject rightControllerObject;
    private GameObject headset;

    public bool leftX;
    public bool leftY;
    public bool leftMenu;
    public bool leftThumbstick;
    public float leftIndexTrigger;
    public float leftHandTrigger;
    public Vector2 leftThumbstickAxes;
    public Vector3 leftLocalPosition;
    public Quaternion leftLocalRotation;

    public bool rightA;
    public bool rightB;
    public bool rightMenu;
    public bool rightThumbstick;
    public float rightIndexTrigger;
    public float rightHandTrigger;
    public Vector2 rightThumbstickAxes;
    public Vector3 rightLocalPosition;
    public Quaternion rightLocalRotation;

    public Vector3 headLocalPosition;
    public Quaternion headLocalRotation;

    public ControllerState(
        OVRInput.Controller leftController,
        OVRInput.Controller rightController,
        GameObject leftControllerObject,
        GameObject rightControllerObject,
        GameObject headset)
    {
        this.leftController = leftController;
        this.rightController = rightController;
        this.leftControllerObject = leftControllerObject;
        this.rightControllerObject = rightControllerObject;
        this.headset = headset;
    }

    public void UpdateState()
    {
        // Left controller state
        this.leftX = OVRInput.Get(OVRInput.RawButton.X, this.leftController);
        this.leftY = OVRInput.Get(OVRInput.RawButton.Y, this.leftController);
        this.leftMenu = OVRInput.Get(OVRInput.RawButton.Start, this.leftController);
        this.leftThumbstick = OVRInput.Get(OVRInput.RawButton.LThumbstick, this.leftController);
        this.leftIndexTrigger = OVRInput.Get(OVRInput.RawAxis1D.LIndexTrigger, this.leftController);
        this.leftHandTrigger = OVRInput.Get(OVRInput.RawAxis1D.LHandTrigger, this.leftController);
        this.leftThumbstickAxes = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick, this.leftController);
        this.leftLocalPosition = OVRInput.GetLocalControllerPosition(this.leftController);
        this.leftLocalRotation = OVRInput.GetLocalControllerRotation(this.leftController);

        // Right controller state
        this.rightA = OVRInput.Get(OVRInput.RawButton.A, this.rightController);
        this.rightB = OVRInput.Get(OVRInput.RawButton.B, this.rightController);
        this.rightMenu = OVRInput.Get(OVRInput.RawButton.Start, this.rightController);
        this.rightThumbstick = OVRInput.Get(OVRInput.RawButton.RThumbstick, this.rightController);
        this.rightIndexTrigger = OVRInput.Get(OVRInput.RawAxis1D.RIndexTrigger, this.rightController);
        this.rightHandTrigger = OVRInput.Get(OVRInput.RawAxis1D.RHandTrigger, this.rightController);
        this.rightThumbstickAxes = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick, this.rightController);
        this.rightLocalPosition = OVRInput.GetLocalControllerPosition(this.rightController);
        this.rightLocalRotation = OVRInput.GetLocalControllerRotation(this.rightController);

        // Headset state
        this.headLocalPosition = headset.transform.position;
        this.headLocalRotation = headset.transform.rotation;
    }

    public override string ToString()
    {
        return $"Left Controller:;" +
               $"  Left X: {leftX};" +
               $"  Left Y: {leftY};" +
               $"  Left Menu: {leftMenu};" +
               $"  Left Thumbstick: {leftThumbstick};" +
               $"  Left Index Trigger: {leftIndexTrigger};" +
               $"  Left Hand Trigger: {leftHandTrigger};" +
               $"  Left Thumbstick Axes: {Vector2ToString(leftThumbstickAxes)};" +
               $"  Left Local Position: {Vector3ToString(leftLocalPosition)};" +
               $"  Left Local Rotation: {QuaternionToString(leftLocalRotation)};" +
               $"|Right Controller:;" +
               $"  Right A: {rightA};" +
               $"  Right B: {rightB};" +
               $"  Right Menu: {rightMenu};" +
               $"  Right Thumbstick: {rightThumbstick};" +
               $"  Right Index Trigger: {rightIndexTrigger};" +
               $"  Right Hand Trigger: {rightHandTrigger};" +
               $"  Right Thumbstick Axes: {Vector2ToString(rightThumbstickAxes)};" +
               $"  Right Local Position: {Vector3ToString(rightLocalPosition)};" +
               $"  Right Local Rotation: {QuaternionToString(rightLocalRotation)};";
    }

    public string ToJSON()
    {
        var leftControllerData = new
        {
            LeftX = leftX,
            LeftY = leftY,
            LeftMenu = leftMenu,
            LeftThumbstick = leftThumbstick,
            LeftIndexTrigger = leftIndexTrigger,
            LeftHandTrigger = leftHandTrigger,
            LeftThumbstickAxes = Vector2ToString(leftThumbstickAxes),
            LeftLocalPosition = Vector3ToString(leftLocalPosition),
            LeftLocalRotation = QuaternionToString(leftLocalRotation)
        };

        var rightControllerData = new
        {
            RightA = rightA,
            RightB = rightB,
            RightMenu = rightMenu,
            RightThumbstick = rightThumbstick,
            RightIndexTrigger = rightIndexTrigger,
            RightHandTrigger = rightHandTrigger,
            RightThumbstickAxes = Vector2ToString(rightThumbstickAxes),
            RightLocalPosition = Vector3ToString(rightLocalPosition),
            RightLocalRotation = QuaternionToString(rightLocalRotation)
        };

        var headPoseData = new { HeadLocalPosition = Vector3ToString(headLocalPosition), HeadLocalRotation = QuaternionToString(headLocalRotation) };

        var data = new
        {
            LeftController = leftControllerData,
            RightController = rightControllerData,
            Headset = headPoseData
        };

        return JsonConvert.SerializeObject(data, Formatting.Indented);
    }




    //// This should be in a utils class. Idk how to get the correct namespace in C# yet.

    public static string Vector2ToString(Vector2 vector)
    {
        return $"{vector.x},{vector.y}";
    }

    public static string Vector3ToString(Vector3 vector)
    {
        return $"{vector.x},{vector.y},{vector.z}";
    }

    public static string QuaternionToString(Quaternion quaternion)
    {
        return $"{quaternion.x},{quaternion.y},{quaternion.z},{quaternion.w}";
    }

}
