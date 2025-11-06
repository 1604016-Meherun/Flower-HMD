using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine.SceneManagement;
using System.Text;

public class SaccadeLogger : MonoBehaviour
{
    // Persist across scenes
    private static List<dataStructure.FrameData> allFrames = new List<dataStructure.FrameData>();
    private string filePath;
    private float startTime;
    public string fileName = "EyeLog";

    [Header("Transforms (Eye + Head)")]
    public Transform leftEyeTransform;
    public Transform rightEyeTransform;
    public Transform centerEyeTransform;

    private readonly string csvHeader = "SceneName,ElapsedTime," +
        "LeftPosX,LeftPosY,LeftPosZ,LeftRotX,LeftRotY,LeftRotZ,LeftRotW,LeftDirX,LeftDirY,LeftDirZ," +
        "RightPosX,RightPosY,RightPosZ,RightRotX,RightRotY,RightRotZ,RightRotW,RightDirX,RightDirY,RightDirZ," +
        "CenterPosX,CenterPosY,CenterPosZ,CenterRotX,CenterRotY,CenterRotZ,CenterRotW,CenterDirX,CenterDirY,CenterDirZ";

    void Awake()
    {
        // Only allow one logger to persist
        if (FindObjectsOfType<SaccadeLogger>().Length > 1)
        {
            Destroy(gameObject);
            return;
        }
        DontDestroyOnLoad(this.gameObject);

        string fileNameNew = $"{fileName}_{DateTime.Now:yyyyMMdd_HHmmss_fff}_{Guid.NewGuid()}.csv";
        filePath = string.IsNullOrEmpty(Application.persistentDataPath)
            ? Path.Combine(Directory.GetCurrentDirectory(), fileNameNew)
            : Path.Combine(Application.persistentDataPath, fileNameNew);

        Debug.Log($"📦 [Awake] Logging to: {filePath}");
    }

    void OnEnable() => SceneManager.sceneUnloaded += HandleSceneUnloaded;
    void OnDisable() => SceneManager.sceneUnloaded -= HandleSceneUnloaded;

    private void HandleSceneUnloaded(Scene scene) => SaveToCSV();

    void Start() => startTime = Time.time;

    void Update()
    {
        if (leftEyeTransform == null || rightEyeTransform == null || centerEyeTransform == null) return;

        float t = Time.time - startTime;
        string scene = SceneManager.GetActiveScene().name;

        // Avoid unnecessary allocations
        allFrames.Add(new dataStructure.FrameData(
            scene,
            t,
            new dataStructure.ObjectData(leftEyeTransform),
            new dataStructure.ObjectData(rightEyeTransform),
            new dataStructure.ObjectData(centerEyeTransform)
        ));
    }

    public void SaveToCSV()
    {
        if (allFrames.Count == 0) return;

        var sb = new StringBuilder();
        sb.AppendLine(csvHeader);
        foreach (var frame in allFrames)
            sb.AppendLine(frame.ToCsvString());

        File.WriteAllText(filePath, sb.ToString());
        Debug.Log($"✅ All scene data saved to {filePath}");
    }
}
