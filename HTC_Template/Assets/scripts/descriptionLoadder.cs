using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class descriptionLoadder : MonoBehaviour
{
    [Header("Buttons")]
    public Button descriptionButton;

    private const string sceneName = "description";
    private const string lastSceneKey = "EyeTrackerreal";
    private bool completed;

    void Start()
    {
        CheckAndHandleCompletion();
    }

    public void CheckAndHandleCompletion()
    {
        string lastScene = PlayerPrefs.GetString(lastSceneKey, "");
        if (!string.IsNullOrEmpty(lastScene) && lastScene == sceneName)
        {
            completed = true;
            PlayerPrefs.DeleteKey(lastSceneKey);
        }

        if (completed)
        {
            descriptionButton.interactable = false;
            descriptionButton.image.color = Color.red;
            // Removed Application.Quit and EditorApplication.isPlaying = false
        }
    }

    public void Description() => ChangeScene(sceneName);

    private void ChangeScene(string sceneName)
    {
        PlayerPrefs.SetString(lastSceneKey, sceneName);
        PlayerPrefs.Save();
        SceneManager.LoadScene(lastSceneKey);
    }
}
