using NetMQ;
using NetMQ.Sockets;
using System;
using System.Collections.Concurrent;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public struct Data
{
    public bool bool_;
    public int int_;
    public string str;
    public byte[] image;
}

public class ReceiverOneWay
{
    private readonly Thread receiveThread;
    private bool running;

    public ReceiverOneWay()
    {
        receiveThread = new Thread((object callback) =>
        {
            using (var socket = new PullSocket())  // <- The PULL socket
            {
                socket.Connect("tcp://localhost:12345");
                while (running)
                {
                    // No socket.SendFrameEmpty(); here
                    string message = socket.ReceiveFrameString();
                    Data data = JsonUtility.FromJson<Data>(message);
                    ((Action<Data>)callback)(data);
                }
            }
        });
    }

    public void Start(Action<Data> callback)
    {
        running = true;
        receiveThread.Start(callback);
    }

    public void Stop()
    {
        running = false;
        receiveThread.Join();
    }
}

public class Client : MonoBehaviour
{
    private readonly ConcurrentQueue<Action> runOnMainThread = new ConcurrentQueue<Action>();
    private ReceiverOneWay receiver;
    private Texture2D tex;
    public RawImage image;

    public void Start()
    {
        tex = new Texture2D(2, 2, TextureFormat.RGB24, mipChain: false);
        image.texture = tex;

        AsyncIO.ForceDotNet.Force();
        // - You might remove it, but if you have more than one socket
        //   in the following threads, leave it.
        receiver = new ReceiverOneWay();
        receiver.Start((Data d) => runOnMainThread.Enqueue(() =>
            {
                Debug.Log(d.str);
                tex.LoadImage(d.image);
            }
        ));
    }

    public void Update()
    {
        if (!runOnMainThread.IsEmpty)
        {
            Action action;
            while (runOnMainThread.TryDequeue(out action))
            {
                action.Invoke();
            }
        }
    }

    private void OnDestroy()
    {
        receiver.Stop();
        NetMQConfig.Cleanup();  // Must be here to work more than once
    }
}