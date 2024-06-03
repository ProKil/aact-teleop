using UnityEngine;
using TMPro;


public class displayCoordinates : MonoBehaviour
{
    public GameObject headset;
    public OVRInput.Controller leftController;
    public OVRInput.Controller rightController;
    public OVRCameraRig cameraRig;
    public TMP_Text coordinateText;
    private ControllerState stateStore;

    // Start is called before the first frame update
    void Start()
    {
        headset = GameObject.Find("CenterEyeAnchor");
        stateStore = new ControllerState(leftController, rightController, headset);
    }

    // Update is called once per frame
    void Update()
    {
        stateStore.UpdateState();

        coordinateText.SetText(
            "Left controller relative: " + stateStore.leftLocalPosition +
            "\nRight controller relative: " + stateStore.rightLocalPosition +
            "\nLeft controller rotation: " + stateStore.leftLocalRotation +
            "\nRight controller rotation: " + stateStore.rightLocalRotation
        );
    }
}