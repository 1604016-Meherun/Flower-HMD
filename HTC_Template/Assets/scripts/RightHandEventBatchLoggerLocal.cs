using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;
using System.Text;
using UnityEngine.Networking;

public class RightHandEventBatchLoggerLocal : MonoBehaviour
{
    [System.Serializable]
    public class ObjectData
    {
        public float posX, posY, posZ, rotX, rotY, rotZ, rotW;

        public ObjectData(Vector3 position, Quaternion rotation)
        {
            posX = position.x;
            posY = position.y;
            posZ = position.z;
            rotX = rotation.x;
            rotY = rotation.y;
            rotZ = rotation.z;
            rotW = rotation.w;
        }

        public string ToCsvString()
        {
            return $"{posX},{posY},{posZ},{rotX},{rotY},{rotZ},{rotW}";
        }
    }

    [System.Serializable]
    public class MovementData
    {
        public ObjectData rightHand, rightRing, rightHead, rightBody;
        public float elapsedTime;
        public int hitRight, inPlayZone, inAZone, inBZone, inCZone,  inRightWireZone;

        public MovementData(float elapsedTime, int inPlayZone, int inAZone, int inBZone, int inCZone, int inRightWireZone, ObjectData rightHand, ObjectData rightRing, ObjectData rightHead, ObjectData rightBody )
        {
            this.elapsedTime = elapsedTime;
            this.inPlayZone = inPlayZone;
            this.inAZone = inAZone;
            this.inBZone = inBZone;
            this.inCZone = inCZone;
            this.inRightWireZone = inRightWireZone;
            this.rightHand = rightHand;
            this.rightRing = rightRing;
            this.rightHead = rightHead;
            this.rightBody = rightBody;
        }

        public string ToCsvString()
        {
            return $"{elapsedTime},{inPlayZone},{inAZone},{inBZone},{inCZone},{inRightWireZone},{rightHand.ToCsvString()},{rightRing.ToCsvString()},{rightHead.ToCsvString()},{rightBody.ToCsvString()}";
        }
    }

    private List<MovementData> movementDataList = new List<MovementData>();
    public Transform rightHand, rightRing, rightHead, rightBody;
    public GameObject rightWire, rightLoop;
    public Collider rightWireZone;

    // public collisionChangeWireZone wireScript;

    private string uniqueFileName = $"RightHandMovementData_{DateTime.Now:yyyyMMdd_HHmmss}.csv".ToLower().Replace(":", "_").Replace("-", "_");
    private string csvHeader = "ElapsedTime,InPlayZone,InAZone,InBZone,InCZone,InWireZone,"+
                               "RightHandPosX,RightHandPosY,RightHandPosZ,RightHandRotX,RightHandRotY,RightHandRotZ,RightHandRotW," +
                               "RingPosX,RingPosY,RingPosZ,RingRotX,RingRotY,RingRotZ,RingRotW," +
                               "HeadPosX,HeadPosY,HeadPosZ,HeadRotX,HeadRotY,HeadRotZ,HeadRotW," +
                               "BodyPosX,BodyPosY,BodyPosZ,BodyRotX,BodyRotY,BodyRotZ,BodyRotW"
                               ;

    public static bool inPlayZoneFlag = false;
    public static bool inAZoneFlag = false;
    public static bool inBZoneFlag = false;
    public static bool inCZoneFlag = false;
    public bool inRightWireZoneFlag = false;
    private float startTimeRight;

    // void Start()
    // {
    //     startTimeRight = Time.time;
    //     Debug.Log($"Data will be saved to: {Path.Combine(Application.persistentDataPath, uniqueFileName)}");
    // }

    void OnEnable()
    {
    startTimeRight = Time.time;
    Debug.Log($"Right Handed Players Data will be saved to: {Path.Combine(Application.persistentDataPath, uniqueFileName)}");
    }

    void Update()
    {
        
        //Debug.Log(Vector3.Distance(GameObject.FindGameObjectWithTag("righthand").transform.position,GameObject.FindGameObjectWithTag("loop_event").transform.position));
        
        float elapsedTime = Time.time - startTimeRight;
        ObjectData rightHandData = new ObjectData(rightHand.position, rightHand.rotation);
        ObjectData rightRingData = new ObjectData(rightRing.position, rightRing.rotation);
        ObjectData rightHeadData = new ObjectData(rightHead.position, rightHead.rotation);
        ObjectData rightBodyData = new ObjectData(rightBody.position, rightBody.rotation);

        int inPlayZone = inPlayZoneFlag ? 1 : 0;
        int inAZone = inAZoneFlag? 1 : 0;
        int inBZone = inBZoneFlag? 1 : 0;
        int inCZone = inCZoneFlag? 1 : 0;
        int inRightWireZone = inRightWireZoneFlag ? 1 : 0;

        MovementData data = new MovementData(elapsedTime, inPlayZone, inAZone, inBZone, inCZone, inRightWireZone, rightHandData, rightRingData, rightHeadData, rightBodyData );
        movementDataList.Add(data);
        
        if (rightWireZone != null && rightRing!= null)
        {
        Renderer rightWireZoneRenderer = rightWireZone.GetComponent<Renderer>();
        Renderer rightRingRenderer = rightRing.GetComponent<Renderer>();
        Color rightWireZoneColor = rightWireZoneRenderer.material.color;
        Color rightRingColor = rightRingRenderer.material.color;
        
        // Check if the color is black
            if (rightWireZoneColor == Color.black && rightRingColor != Color.black)
            {
            inRightWireZoneFlag = true;
            // Debug.Log("Wire is black, inWireZoneFlag set to true");
            }
            else
            {
            inRightWireZoneFlag = false;
            // Debug.Log("Wire is red, inWireZoneFlag set to false");
            }
        }
    }

    

    public void inPlayZoneFlagTrue()// Actually it is player in Play Zone
    {
        inPlayZoneFlag = true;
        Debug.Log("inPlayZoneFlagUpdatedTrue");
    }  

    public void inPlayZoneFlagFalse()// Actually it is player is not in Play Zone
    {
        inPlayZoneFlag = false;
        Debug.Log("inPlayZoneFlagUpdatedFalse");
    }

    public void inAZoneFlagTrue()// Actually it is player in Play Zone
    {
        inAZoneFlag = true;
        Debug.Log("inAZoneFlagUpdatedTrue");
    }  

    public void inAZoneFlagFalse()// Actually it is player is not in Play Zone
    {
        inAZoneFlag = false;
        Debug.Log("inAZoneFlagFlagUpdatedFalse");
    }

    public void inBZoneFlagTrue()// Actually it is player in Play Zone
    {
        inBZoneFlag = true;
        Debug.Log("inBZoneFlagUpdatedTrue");
    }  

    public void inBZoneFlagFalse()// Actually it is player is not in Play Zone
    {
        inBZoneFlag = false;
        Debug.Log("inBZoneFlagFlagUpdatedFalse");
    }

    public void inCZoneFlagTrue()// Actually it is player in Play Zone
    {
        inCZoneFlag = true;
        Debug.Log("inCZoneFlagUpdatedTrue");
    }  

    public void inCZoneFlagFalse()// Actually it is player is not in Play Zone
    {
        inCZoneFlag = false;
        Debug.Log("inCZoneFlagFlagUpdatedFalse");
    }


    // void OnTriggerStay(Collider other)
    // {
    //     if (gameObject == wireZone && other.gameObject == wire)
    //     {
    //         inWireZoneFlag = true;
    //         Debug.Log("WireZoneFlagUpdatedTrue");
    //     }
    // }


    public void OnApplicationQuit()
    {
        WriteAllDataToFile();
        // SendDataToELKServer();
        // this.enabled = false; 
        // Debug.Log("RightLoggerScript has been disabled on application quit.");
    }

    void WriteAllDataToFile()
    {
        string filePath = Path.Combine(Application.persistentDataPath, uniqueFileName);
        using (StreamWriter streamWriter = new StreamWriter(filePath))
        {
            streamWriter.WriteLine(csvHeader);
            foreach (var data in movementDataList)
            {
                streamWriter.WriteLine(data.ToCsvString());
            }
        }
        Debug.Log($"Successfully saved all data to {uniqueFileName}");
    }

    // private string baseUrl = "https://erbium.host.ualr.edu:9200";
    // private string username = "elastic";
    // private string password = "ovVeWTnmYQrAF9XOJCAA";

    // [System.Serializable]
    // public class BulkData
    // {
    //     public List<MovementData> movements;
    // }

    // void SendDataToELKServer()
    // {
    //     BulkData bulkData = new BulkData { movements = new List<MovementData>(movementDataList) };
    //     string json = JsonUtility.ToJson(bulkData);

    //     // Send data synchronously
    //     using (UnityWebRequest request = new UnityWebRequest(baseUrl + $"/{uniqueFileName}/_doc", "POST"))
    //     {
    //         byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
    //         request.uploadHandler = new UploadHandlerRaw(bodyRaw);
    //         request.downloadHandler = new DownloadHandlerBuffer();

    //         request.SetRequestHeader("Content-Type", "application/json");
    //         string auth = Convert.ToBase64String(Encoding.GetEncoding("ISO-8859-1").GetBytes(username + ":" + password));
    //         request.SetRequestHeader("Authorization", "Basic " + auth);

    //         var asyncOperation = request.SendWebRequest();

    //         while (!asyncOperation.isDone)
    //         {
    //             Debug.Log($"Elk - Sending the data");
    //         }

    //         if (request.result != UnityWebRequest.Result.Success)
    //         {
    //             Debug.LogError($"Error sending data to ELK: {request.error}");
    //         }
    //         else
    //         {
    //             Debug.Log("Successfully sent data to Elasticsearch in bulk");
    //         }
    //     }
    // }


}
