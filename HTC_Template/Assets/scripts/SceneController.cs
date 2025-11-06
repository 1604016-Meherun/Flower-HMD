using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class SceneController : MonoBehaviour
{
    public float sceneDuration = 5f; // 2 minutes per scene
    private float timer = 0f;

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= sceneDuration)
        {
            LoadNextScene();
        }
    }

    void LoadNextScene()
    {
        int currentSceneIndex = SceneManager.GetActiveScene().buildIndex;
        int nextSceneIndex = (currentSceneIndex + 1) % SceneManager.sceneCountInBuildSettings;

        // Load next scene asynchronously for smooth transitions
        SceneManager.LoadSceneAsync(nextSceneIndex);
    }

    // public string[] sceneNames = { "fixation", "saccade", "smooth_pursuit" }; // Scene names
    // public float sceneDuration = 120f; // 2 minutes per scene
    // public GameObject instructionPanel; // UI Panel to show messages
    // public Text instructionText; // UI Text to display messages
    // public Button continueButton; // UI Button to proceed
    // public Canvas canvas; // Reference to the Canvas

    // private int currentSceneIndex = 0;
    // private bool userAgreed = false;

    // void Start()
    // {
    //     // Ensure UI elements are properly assigned
    //     if (instructionPanel == null || instructionText == null || continueButton == null || canvas == null)
    //     {
    //         Debug.LogError("Assign UI elements in the Inspector!");
    //         return;
    //     }

    //     // Set the Canvas to World Space
    //     canvas.renderMode = RenderMode.WorldSpace;

    //     // Position the Canvas in front of the camera
    //     canvas.transform.position = Camera.main.transform.position + Camera.main.transform.forward * 2f;
    //     canvas.transform.rotation = Camera.main.transform.rotation;

    //     // Start the first scene transition process
    //     StartCoroutine(HandleSceneTransitions());
    // }

    // IEnumerator HandleSceneTransitions()
    // {
    //     while (currentSceneIndex < sceneNames.Length)
    //     {
    //         // Show preparation message
    //         ShowPreparationMessage(sceneNames[currentSceneIndex]);

    //         // Wait for user agreement
    //         userAgreed = false;
    //         continueButton.onClick.RemoveAllListeners(); // Clear previous listeners
    //         continueButton.onClick.AddListener(() => userAgreed = true);
    //         yield return new WaitUntil(() => userAgreed);

    //         // Load the scene
    //         SceneManager.LoadScene(sceneNames[currentSceneIndex]);

    //         // Wait for the scene duration
    //         yield return new WaitForSeconds(sceneDuration);

    //         // Move to the next scene
    //         currentSceneIndex++;
    //     }

    //     // When all scenes are completed, show a final message or return to the main menu
    //     instructionText.text = "Thank you! The experiment is over.";
    //     instructionPanel.SetActive(true);
    // }

    // void ShowPreparationMessage(string sceneName)
    // {
    //     switch (sceneName)
    //     {
    //         case "fixation":
    //             instructionText.text = "You have to fix your eyes on the ball. Are you ready?";
    //             break;
    //         case "saccade":
    //             instructionText.text = "You need to search the ball. Are you ready?";
    //             break;
    //         case "smooth_pursuit":
    //             instructionText.text = "You need to follow the ball. Are you ready?";
    //             break;
    //         default:
    //             instructionText.text = "Get ready for the next scene. Are you ready?";
    //             break;
    //     }
    //     instructionPanel.SetActive(true);
    // }
}
