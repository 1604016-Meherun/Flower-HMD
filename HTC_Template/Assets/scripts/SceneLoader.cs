using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Collections.Generic;

public class SceneLoader : MonoBehaviour
{
    [Header("Buttons")]
    public Button fixationButton;
    public Button smoothPursuitButton;
    public Button saccadeButton;

    private static readonly string[] sceneNames = { "Fixation", "smoothpursuit", "saccade" };
    private static readonly string lastSceneKey = "LastScene";
    private static HashSet<string> completedScenes = new HashSet<string>();

    void Start()
    {
        // Map buttons to scene names
        var sceneButtonMap = new Dictionary<string, Button>(3)
        {
            { sceneNames[0], fixationButton },
            { sceneNames[1], smoothPursuitButton },
            { sceneNames[2], saccadeButton }
        };

        // Load last completed scene only if not already in set
        string lastScene = PlayerPrefs.GetString(lastSceneKey, "");
        if (!string.IsNullOrEmpty(lastScene) && completedScenes.Add(lastScene))
        {
            PlayerPrefs.DeleteKey(lastSceneKey);
        }

        // Disable completed scene buttons and color red
        foreach (var name in sceneNames)
        {
            if (completedScenes.Contains(name))
            {
                var btn = sceneButtonMap[name];
                btn.interactable = false;
                btn.image.color = Color.red;
            }
        }

        // If all scenes completed, clear and quit
        if (completedScenes.Count == sceneNames.Length)
        {
            PlayerPrefs.DeleteAll();
            completedScenes.Clear();

#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }

    public void sceneChanger_fixation() => ChangeScene(sceneNames[0]);
    public void sceneChanger_smooth_pursuit() => ChangeScene(sceneNames[1]);
    public void sceneChanger_saccade() => ChangeScene(sceneNames[2]);

    private void ChangeScene(string sceneName)
    {
        PlayerPrefs.SetString(lastSceneKey, sceneName);
        PlayerPrefs.Save();
        SceneManager.LoadScene(sceneName);
    }
}
