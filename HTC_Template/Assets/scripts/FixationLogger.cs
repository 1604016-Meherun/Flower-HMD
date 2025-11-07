using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine.SceneManagement;

public class FixationLogger : MonoBehaviour
{
    private List<dataStructure.FrameData> allFrames;
    private string filePath;
    private float startTime;
    public string fileName;

    public FederatedTrainer trainer;   // ✅ added: we’ll send JSON directly to Python via this

    [Header("Transforms (Eye + Head)")]
    public Transform leftEyeTransform;
    public Transform rightEyeTransform;
    public Transform centerEyeTransform;

    void Awake()
    {
        if (FindObjectsOfType<FixationLogger>().Length > 1)
        {
            Destroy(gameObject);
            return;
        }
        DontDestroyOnLoad(gameObject);

        if (string.IsNullOrEmpty(fileName))
            fileName = "EyeLog";

        string guid = Guid.NewGuid().ToString();
        string fileNameNew = $"{fileName}_{DateTime.Now:yyyyMMdd_HHmmss}_{guid}.json";

        filePath = string.IsNullOrEmpty(Application.persistentDataPath)
            ? Path.Combine(Directory.GetCurrentDirectory(), fileNameNew)
            : Path.Combine(Application.persistentDataPath, fileNameNew);

#if UNITY_EDITOR || DEVELOPMENT_BUILD
        Debug.Log($"📦 [Awake] Logging to: {filePath}");
#endif

        allFrames = new List<dataStructure.FrameData>(10000);
    }

    void OnEnable() => SceneManager.sceneUnloaded += OnSceneUnloaded;
    void OnDisable() => SceneManager.sceneUnloaded -= OnSceneUnloaded;

    private void OnSceneUnloaded(Scene scene) => SaveToJSON();

    void Start()
    {
        // ✅ ensure we have a trainer instance
        if (trainer == null) trainer = FindObjectOfType<FederatedTrainer>();

        startTime = Time.time;
#if UNITY_EDITOR || DEVELOPMENT_BUILD
        Debug.Log($"📦 Logging started at: {filePath}");
#endif
    }

    void Update()
    {
        if (!leftEyeTransform || !rightEyeTransform || !centerEyeTransform) return;

        float t = Time.time - startTime;
        string scene = SceneManager.GetActiveScene().name;

        allFrames.Add(new dataStructure.FrameData(
            scene,
            t,
            new dataStructure.ObjectData(leftEyeTransform),
            new dataStructure.ObjectData(rightEyeTransform),
            new dataStructure.ObjectData(centerEyeTransform)
        ));
    }

    public void SaveToJSON()
    {
        if (allFrames.Count == 0) return;

        try
        {
            // Wrap all data in a container to serialize properly
            var wrapper = new FrameDataWrapper { frames = allFrames };

            string json = JsonUtility.ToJson(wrapper, true); // pretty print

            // (Optional) keep a debug copy on disk. Remove these two lines if you don't want any files.
            // File.WriteAllText(filePath, json);
            // Debug.Log($"✅ Session JSON written to {filePath}");

            // ✅ DATA-ONLY: send the JSON string to Python (no file path)
            if (trainer != null)
            {
                Debug.Log("[FL] Launching Flower client (stdin, data-only) ...");
                trainer.TrainOnJsonString(json);   // <-- this is the only behavioral change
            }
            else
            {
                Debug.LogWarning("[FL] FederatedTrainer not found in scene.");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"❌ Failed to save/send JSON: {ex.Message}");
        }
    }

    [Serializable]
    private class FrameDataWrapper
    {
        public List<dataStructure.FrameData> frames;
    }
}
