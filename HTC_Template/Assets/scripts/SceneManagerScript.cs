using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneManagerScript : MonoBehaviour
{
    public void GoToSceneTwo()
    {
        SceneManager.LoadScene("smooth_pursuit");
    }



//     public float sceneDuration = 5f; // 5 seconds
//     private float timer = 0f;

//     void Update()
//     {
//         timer += Time.deltaTime;

//         if (timer >= sceneDuration)
//         {
//             LoadNextScene();
//         }
//     }

//     void LoadNextScene()
//     {
//         int currentSceneIndex = SceneManager.GetActiveScene().buildIndex;
//         int nextSceneIndex = (currentSceneIndex + 1) % SceneManager.sceneCountInBuildSettings;
//         if (nextSceneIndex != currentSceneIndex) // Ensure the next scene is different
//         {
//             SceneManager.LoadScene(nextSceneIndex);
//         }
//     }

}
