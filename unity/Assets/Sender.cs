using NetMQ;
using NetMQ.Sockets;
using System;
using System.Collections.Concurrent;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;
using AsyncIO;

public class Sender : MonoBehaviour
{
    private DateTime today;
    private PushSocket socket;

    public OVRInput.Controller leftController;
    public OVRInput.Controller rightController;
    public GameObject leftControllerObject;
    public GameObject rightControllerObject;
    public GameObject headset;
    private ControllerState stateStore;

    private void Start()
    {
        ForceDotNet.Force();
        socket = new PushSocket();
        socket.Bind("tcp://*:12345");
        stateStore = new ControllerState(
            leftController,
            rightController,
            leftControllerObject,
            rightControllerObject,
            headset
        );
    }

    private void Update()
    {
        stateStore.UpdateState();
        socket.SendFrame(stateStore.ToJSON());
    }

    private void OnDestroy()
    {
        var terminationString = "terminate";
        for (int i = 0; i < 10; i++)
        {
            socket.SendFrame(terminationString);
        }
        socket.Close();
        NetMQConfig.Cleanup();
    }
    private void OnApplicationPause()
    {
        var terminationString = "terminate";
        for (int i = 0; i < 10; i++)
        {
            socket.SendFrame(terminationString);
        }
        socket.Close();
        NetMQConfig.Cleanup();
    }
}
