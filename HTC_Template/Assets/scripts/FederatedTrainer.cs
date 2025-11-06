// using System;
// using System.Diagnostics;
// using System.IO;
// using System.Text;
// using UnityEngine;
// using Debug = UnityEngine.Debug;

// public class FederatedTrainer : MonoBehaviour
// {
//     [Header("Paths (set in Inspector)")]
//     public string pythonExe   = @"C:\Users\mshraboni-local\AppData\Local\Programs\Python\Python38\python.exe"; // <-- POINT THIS TO YOUR WORKING VENV
//     public string clientScript = @"D:\Projects\Flower+HMD\FL+Flwr+VRData\flwr_time_series_client.py";

//     [Header("Flower Server")]
//     public string serverAddress = "127.0.0.1:5006";

//     private Process proc;

//     void Awake()
//     {
//         Debug.Log("[FL] FederatedTrainer present");
//         Debug.Log("[FL] pythonExe: " + pythonExe);
//         Debug.Log("[FL] clientScript: " + clientScript);
//         Debug.Log("[FL] serverAddress: " + serverAddress);
//         DontDestroyOnLoad(gameObject);
//     }

//     public void TrainOnCsv(string csvFullPath)
//     {
//         Debug.Log("[FL] TrainOnCsv: " + csvFullPath);

//         if (!File.Exists(csvFullPath))
//         {
//             Debug.LogError("[FL] CSV file not found: " + csvFullPath);
//             return;
//         }
//         if (!File.Exists(pythonExe))
//         {
//             Debug.LogError("[FL] python.exe not found: " + pythonExe);
//             return;
//         }
//         if (!File.Exists(clientScript))
//         {
//             Debug.LogError("[FL] client script not found: " + clientScript);
//             return;
//         }

//         string args = $"\"{clientScript}\" --csv \"{csvFullPath}\" --server \"{serverAddress}\"";
//         string workdir = Path.GetDirectoryName(clientScript);

//         var psi = new ProcessStartInfo
//         {
//             FileName = pythonExe,
//             Arguments = args,
//             WorkingDirectory = string.IsNullOrEmpty(workdir) ? Environment.CurrentDirectory : workdir,
//             UseShellExecute = false,
//             RedirectStandardOutput = true,
//             RedirectStandardError  = true,
//             CreateNoWindow = true,
//             StandardOutputEncoding = Encoding.UTF8,
//             StandardErrorEncoding  = Encoding.UTF8,
//         };

//         try
//         {
//             proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
//             proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Debug.Log("[FL/PY] " + e.Data); };
//             proc.ErrorDataReceived  += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Debug.LogWarning("[FL/PY-ERR] " + e.Data); };
//             proc.Exited += (_, __) =>
//             {
//                 try { Debug.Log("[FL] Python exited, code " + proc.ExitCode); } catch {}
//             };

//             Debug.Log("[FL] Starting Python:");
//             Debug.Log("[FL]   exe : " + psi.FileName);
//             Debug.Log("[FL]   args: " + psi.Arguments);
//             Debug.Log("[FL]   cwd : " + psi.WorkingDirectory);

//             if (proc.Start())
//             {
//                 proc.BeginOutputReadLine();
//                 proc.BeginErrorReadLine();
//                 Debug.Log("[FL] Started Flower client");
//             }
//             else
//             {
//                 Debug.LogError("[FL] Failed to start Python (proc.Start returned false)");
//             }
//         }
//         catch (Exception ex)
//         {
//             Debug.LogError("[FL] Exception starting Python: " + ex.Message + "\n" + ex.StackTrace);
//         }
//     }

//     void OnApplicationQuit()
//     {
//         try
//         {
//             if (proc != null && !proc.HasExited)
//             {
//                 // Optional: comment this out if you want training to continue after Editor stops.
//                 proc.Kill();
//             }
//         }
//         catch { }
//         proc = null;
//     }
// }


using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class FederatedTrainer : MonoBehaviour
{
    [Header("Paths (set in Inspector)")]
    public string pythonExe    = @"C:\Users\mshraboni-local\AppData\Local\Programs\Python\Python38\python.exe";
    public string clientScript = @"D:\Projects\Flower+HMD\Flower+HMD\FL+Flwr+VRData\flwr_time_series_client.py";

    [Header("Flower Server")]
    public string serverAddress = "127.0.0.1:5006";

    private Process proc;

    void Awake()
    {
        Debug.Log("[FL] FederatedTrainer present");
        Debug.Log("[FL] pythonExe: " + pythonExe);
        Debug.Log("[FL] clientScript: " + clientScript);
        Debug.Log("[FL] serverAddress: " + serverAddress);
        DontDestroyOnLoad(gameObject);
    }

    // ---- CSV entry point (kept for backward compatibility) ----
    public void TrainOnCsv(string csvFullPath)
    {
        if (!Preflight(csvFullPath)) return;
        string args = $"\"{clientScript}\" --path \"{csvFullPath}\" --server \"{serverAddress}\"";
        StartPython(args);
    }

    // ---- NEW: JSON entry point ----
    public void TrainOnJson(string jsonFullPath)
    {
        if (!Preflight(jsonFullPath)) return;
        // If your Python client expects "--format json" instead, change the args line accordingly.
        string args = $"\"{clientScript}\" --path \"{jsonFullPath}\" --server \"{serverAddress}\"";
        StartPython(args);
    }

    // ---- Shared helpers ----
    private bool Preflight(string dataPath)
    {
        Debug.Log("[FL] Data path: " + dataPath);

        if (!File.Exists(dataPath))
        {
            Debug.LogError("[FL] Data file not found: " + dataPath);
            return false;
        }
        if (string.IsNullOrWhiteSpace(pythonExe) || !File.Exists(pythonExe))
        {
            Debug.LogError("[FL] python.exe not found: " + pythonExe);
            return false;
        }
        if (string.IsNullOrWhiteSpace(clientScript) || !File.Exists(clientScript))
        {
            Debug.LogError("[FL] client script not found: " + clientScript);
            return false;
        }
        return true;
    }

    private void StartPython(string args)
    {
        string workdir = Path.GetDirectoryName(clientScript);

        var psi = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = args,
            WorkingDirectory = string.IsNullOrEmpty(workdir) ? Environment.CurrentDirectory : workdir,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding  = Encoding.UTF8,
        };

        try
        {
            proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Debug.Log("[FL/PY] " + e.Data); };
            proc.ErrorDataReceived  += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Debug.LogWarning("[FL/PY-ERR] " + e.Data); };
            proc.Exited += (_, __) =>
            {
                try { Debug.Log("[FL] Python exited, code " + proc.ExitCode); } catch {}
            };

            Debug.Log("[FL] Starting Python:");
            Debug.Log("[FL]   exe : " + psi.FileName);
            Debug.Log("[FL]   args: " + psi.Arguments);
            Debug.Log("[FL]   cwd : " + psi.WorkingDirectory);

            if (proc.Start())
            {
                proc.BeginOutputReadLine();
                proc.BeginErrorReadLine();
                Debug.Log("[FL] Started Flower client");
            }
            else
            {
                Debug.LogError("[FL] Failed to start Python (proc.Start returned false)");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("[FL] Exception starting Python: " + ex.Message + "\n" + ex.StackTrace);
        }
    }

    void OnApplicationQuit()
    {
        try
        {
            if (proc != null && !proc.HasExited)
            {
                // Optional: comment this out if you want training to continue after Editor stops.
                proc.Kill();
            }
        }
        catch { }
        proc = null;
    }
}
