using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;

public class DataLogger : MonoBehaviour
{
   [System.Serializable]
    public class ObjectData
    {
        public float posX, posY, posZ, dirX, dirY, dirZ;

        public ObjectData(Vector3 position, Vector3 direction)
        {
            posX = position.x;
            posY = position.y;
            posZ = position.z;
            dirX = direction.x;
            dirY = direction.y;
            dirZ = direction.z;
        }

        public string ToCsvString()
        {
            return $"{posX},{posY},{posZ},{dirX},{dirY},{dirZ}";
        }
    }

    [System.Serializable]
    public class GazeData
    {
        public float elapsedTime;
        public ObjectData leftEye, rightEye, centerEye;

        public GazeData(float elapsedTime, ObjectData leftEye, ObjectData rightEye, ObjectData centerEye)
        {
            this.elapsedTime = elapsedTime;
            this.leftEye = leftEye;
            this.rightEye = rightEye;
            this.centerEye = centerEye;
        }

        public string ToCsvString()
        {
            return $"{elapsedTime:F3},{leftEye.ToCsvString()},{rightEye.ToCsvString()},{centerEye.ToCsvString()}";
        }
    }

    private List<GazeData> gazeDataList = new();
    private string uniqueFileName;
    private string csvHeader = "Time," +
                               "LeftPosX,LeftPosY,LeftPosZ,LeftDirX,LeftDirY,LeftDirZ," +
                               "RightPosX,RightPosY,RightPosZ,RightDirX,RightDirY,RightDirZ," +
                               "CenterPosX,CenterPosY,CenterPosZ,CenterDirX,CenterDirY,CenterDirZ,";

    [Header("Transforms")]
    public Transform leftEye, rightEye, centerEye;

    private float startTime;

    void OnEnable()
    {
        startTime = Time.time;
        uniqueFileName = $"EyeGazeData_{DateTime.Now:yyyyMMdd_HHmmss}.csv";
        Debug.Log($"🧠 Eye gaze data will be saved to: {Path.Combine(Application.persistentDataPath, uniqueFileName)}");
    }

    void Update()
    {
        if (!leftEye || !rightEye || !centerEye ) return;

        float elapsedTime = Time.time - startTime;

        ObjectData left = new(leftEye.position, leftEye.forward);
        ObjectData right = new(rightEye.position, rightEye.forward);
        ObjectData center = new(centerEye.position, centerEye.forward); 

        gazeDataList.Add(new GazeData(elapsedTime, left, right, center));
    }

    public void SaveEyeGazeData()
    {
        string filePath = Path.Combine(Application.persistentDataPath, uniqueFileName);
        using (StreamWriter writer = new StreamWriter(filePath))
        {
            writer.WriteLine(csvHeader);
            foreach (var data in gazeDataList)
                writer.WriteLine(data.ToCsvString());
        }

        Debug.Log($"✅ Eye tracking data saved to: {filePath}");
    }

    void OnApplicationQuit()
    {
        Debug.Log($"✅ Application Quit");
        SaveEyeGazeData();  // fallback
    }
}
