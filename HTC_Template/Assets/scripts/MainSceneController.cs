using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MainSceneController : MonoBehaviour
{
    public GameObject Plane; // Assign the existing Plane in the Inspector
    public float displayDuration = 15f; // Time before switching to the next scene

    void Start()
    {
        if (Plane == null)
        {
            Debug.LogError("Plane is not assigned in the Inspector!");
            return;
        }

        // Ensure the Plane is active
        Plane.SetActive(true);

        // Start the coroutine to switch the scene after the duration
        StartCoroutine(SwitchSceneAfterDelay());
    }

    IEnumerator SwitchSceneAfterDelay()
    {
        // Wait for the specified duration
        yield return new WaitForSeconds(displayDuration);

        // Load the next scene (assuming the next scene is indexed at 1)
        SceneManager.LoadScene("saccade");
    }
}
