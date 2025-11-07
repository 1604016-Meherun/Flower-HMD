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
        DontDestroyOnLoad(gameObject);
    }
    // ➊ NEW: data-only entry point (no file path)
    public void TrainOnJsonString(string jsonData)
    {
        if (string.IsNullOrWhiteSpace(jsonData))
        {
            Debug.LogError("[FL] Empty JSON payload");
            return;
        }
        if (string.IsNullOrWhiteSpace(pythonExe) || !File.Exists(pythonExe))
        {
            Debug.LogError("[FL] python.exe not found: " + pythonExe);
            return;
        }
        if (string.IsNullOrWhiteSpace(clientScript) || !File.Exists(clientScript))
        {
            Debug.LogError("[FL] client script not found: " + clientScript);
            return;
        }

        // Run client in stdin mode (no path)
        string args = $"\"{clientScript}\" --stdin --server \"{serverAddress}\"";
        StartPythonWithStdin(args, jsonData);
    }

    // ➋ NEW: like StartPython, but also writes to stdin
    private void StartPythonWithStdin(string args, string stdinPayload)
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
            RedirectStandardInput  = true,   // <-- IMPORTANT
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding  = Encoding.UTF8,
        };

        try
        {
            proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Debug.Log("[FL/PY] " + e.Data); };
            proc.ErrorDataReceived  += (_, e) => { if (!string.IsNullOrEmpty(e.Data)) Debug.LogWarning("[FL/PY-ERR] " + e.Data); };
            proc.Exited += (_, __) => { try { Debug.Log("[FL] Python exited, code " + proc.ExitCode); } catch {} };

            Debug.Log("[FL] Starting Python (stdin mode):");
            Debug.Log("[FL]   exe : " + psi.FileName);
            Debug.Log("[FL]   args: " + psi.Arguments);
            Debug.Log("[FL]   cwd : " + psi.WorkingDirectory);

            if (proc.Start())
            {
                proc.BeginOutputReadLine();
                proc.BeginErrorReadLine();

                // Write the JSON payload to stdin then close
                proc.StandardInput.Write(stdinPayload);
                proc.StandardInput.Close();

                Debug.Log("[FL] Sent JSON to Python via stdin");
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
        try { if (proc != null && !proc.HasExited) proc.Kill(); } catch {}
        proc = null;
    }
}
