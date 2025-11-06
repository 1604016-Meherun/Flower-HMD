using UnityEngine;
using UnityEngine.SceneManagement; // Import SceneManagement

public class SceneSwitcher : MonoBehaviour
{
    // Method to change scene using the scene name
    public void ChangeSceneByName(string sceneName)
    {
        SceneManager.LoadScene(sceneName);
    }

    // Method to change scene using the scene index
    public void ChangeSceneByIndex(int sceneIndex)
    {
        SceneManager.LoadScene(sceneIndex);
    }

    // Async scene loading (for smooth transition)
    public void ChangeSceneAsync(string sceneName)
    {
        // StartCoroutine(LoadSceneAsync(sceneName));
    }

    // private IEnumerator LoadSceneAsync(string sceneName)
    // {
    //     AsyncOperation asyncLoad = SceneManager.LoadSceneAsync(sceneName);
    //     while (!asyncLoad.isDone)
    //     {
    //         yield return null;
    //     }
    // }
}
