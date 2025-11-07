// using UnityEngine;
// using System.IO;

// public class FLKick : MonoBehaviour
// {
//     public FederatedTrainer trainer;  
//     // Paste a real JSON path here:
//     public string jsonPath = @"C:/Users/mshraboni-local/AppData/LocalLow/DefaultCompany/HTC_Template/Fixation_20251106_144318_dd6fa170-870f-4b1f-8048-67ec1928da47.json";  

//     void Start()
//     {
//         Debug.Log("[FL] FLKick.Start()");

//         if (trainer == null)
//             trainer = FindObjectOfType<FederatedTrainer>();

//         if (trainer == null)
//         {
//             Debug.LogError("[FL] No FederatedTrainer in scene!");
//             return;
//         }

//         if (string.IsNullOrEmpty(jsonPath) || !File.Exists(jsonPath))
//         {
//             Debug.LogError("[FL] JSON file not found: " + jsonPath);
//             return;
//         }

//         Debug.Log("[FL] Kicking training manually (JSON mode)...");
//         trainer.TrainOnJson(jsonPath);
//     }
// }
